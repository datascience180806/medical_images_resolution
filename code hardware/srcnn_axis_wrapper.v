`timescale 1ns / 1ps

// AXI4-Stream wrapper for one 128x128 grayscale patch.
// Input and output byte order is little endian inside each 32-bit word:
// byte 0 -> tdata[7:0], byte 1 -> [15:8], byte 2 -> [23:16], byte 3 -> [31:24].
module srcnn_axis_wrapper #(
    parameter C_PIXEL_WIDTH         = 8,
    parameter C_TDATA_WIDTH         = 32,
    parameter C_FIFO_DEPTH          = 16,
    parameter C_AF_MARGIN           = 4,
    parameter C_CORE_LATENCY        = 21,
    parameter C_S_AXIS_TDATA_WIDTH  = 32,
    parameter C_M_AXIS_TDATA_WIDTH  = 32,
    parameter IMAGE_WIDTH           = 128,
    parameter IMAGE_HEIGHT          = 128,
    parameter WORDS_PER_PATCH       = 4096,
    parameter FIFO_DEPTH            = 16,
    parameter FIFO_ADDRESS_BITS     = 4,
    parameter OUTPUT_ZERO_POINT     = 128,
    parameter WEIGHT_FILE           = "weights_hex_clean.txt",
    parameter BIAS_FILE             = "biases_hex_clean.txt"
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [31:0]          s_axis_tdata,
    input  wire                 s_axis_tvalid,
    output wire                 s_axis_tready,
    input  wire                 s_axis_tlast,

    output wire [31:0]          m_axis_tdata,
    output wire                 m_axis_tvalid,
    input  wire                 m_axis_tready,
    output wire                 m_axis_tlast,

    output reg                  protocol_error
);

    reg [31:0] word_buffer;
    reg        word_buffer_valid;
    reg [1:0]  byte_index;
    reg [12:0] input_word_count;
    reg        frame_input_complete;

    wire [7:0] selected_input_byte;
    reg  [7:0] selected_input_byte_reg;
    always @* begin
        case (byte_index)
            2'd0: selected_input_byte_reg = word_buffer[7:0];
            2'd1: selected_input_byte_reg = word_buffer[15:8];
            2'd2: selected_input_byte_reg = word_buffer[23:16];
            default: selected_input_byte_reg = word_buffer[31:24];
        endcase
    end
    assign selected_input_byte = selected_input_byte_reg;

    wire signed [8:0] centered_pixel_wide;
    wire signed [7:0] centered_pixel;
    assign centered_pixel_wide = $signed({1'b0, selected_input_byte}) - 9'sd128;
    assign centered_pixel = centered_pixel_wide[7:0];

    wire       core_in_ready;
    wire       core_out_valid;
    wire       core_out_ready;
    wire [7:0] core_out_pixel;
    wire       core_input_transfer;
    wire       core_output_transfer;

    assign s_axis_tready = !word_buffer_valid && !frame_input_complete;
    assign core_input_transfer = word_buffer_valid && core_in_ready;

    srcnn_top_core #(
        .IMAGE_WIDTH       (IMAGE_WIDTH),
        .IMAGE_HEIGHT      (IMAGE_HEIGHT),
        .OUTPUT_ZERO_POINT (OUTPUT_ZERO_POINT),
        .WEIGHT_FILE       (WEIGHT_FILE),
        .BIAS_FILE         (BIAS_FILE)
    ) core (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (word_buffer_valid),
        .in_ready  (core_in_ready),
        .in_pixel  (centered_pixel),
        .out_valid (core_out_valid),
        .out_ready (core_out_ready),
        .out_pixel (core_out_pixel)
    );

    reg [31:0] pack_data;
    reg [1:0]  pack_byte_count;
    reg [12:0] pushed_word_count;

    reg [31:0] fifo_data [0:FIFO_DEPTH-1];
    reg        fifo_last [0:FIFO_DEPTH-1];
    reg [FIFO_ADDRESS_BITS-1:0] fifo_write_pointer;
    reg [FIFO_ADDRESS_BITS-1:0] fifo_read_pointer;
    reg [FIFO_ADDRESS_BITS:0]   fifo_count;

    wire fifo_empty;
    wire fifo_full;
    wire fifo_pop;
    wire fifo_can_push;
    wire fifo_push;

    assign fifo_empty = (fifo_count == 0);
    assign fifo_full  = (fifo_count == FIFO_DEPTH);
    assign m_axis_tvalid = !fifo_empty;
    assign m_axis_tdata  = fifo_data[fifo_read_pointer];
    assign m_axis_tlast  = fifo_last[fifo_read_pointer];
    assign fifo_pop = m_axis_tvalid && m_axis_tready;
    assign fifo_can_push = !fifo_full || fifo_pop;

    assign core_out_ready = (pack_byte_count != 3) || fifo_can_push;
    assign core_output_transfer = core_out_valid && core_out_ready;
    assign fifo_push = core_output_transfer && (pack_byte_count == 3);

    always @(posedge clk) begin
        if (!rst_n) begin
            word_buffer          <= 32'd0;
            word_buffer_valid    <= 1'b0;
            byte_index           <= 2'd0;
            input_word_count     <= 13'd0;
            frame_input_complete <= 1'b0;
            protocol_error       <= 1'b0;
        end else begin
            if (s_axis_tvalid && s_axis_tready) begin
                word_buffer       <= s_axis_tdata;
                word_buffer_valid <= 1'b1;
                byte_index        <= 2'd0;

                if (s_axis_tlast != (input_word_count == WORDS_PER_PATCH - 1))
                    protocol_error <= 1'b1;

                if (input_word_count == WORDS_PER_PATCH - 1) begin
                    frame_input_complete <= 1'b1;
                end else begin
                    input_word_count <= input_word_count + 1'b1;
                end
            end

            if (core_input_transfer) begin
                if (byte_index == 3) begin
                    byte_index        <= 2'd0;
                    word_buffer_valid <= 1'b0;
                end else begin
                    byte_index <= byte_index + 1'b1;
                end
            end

            if (fifo_pop && m_axis_tlast) begin
                frame_input_complete <= 1'b0;
                input_word_count     <= 13'd0;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            pack_data          <= 32'd0;
            pack_byte_count    <= 2'd0;
            pushed_word_count  <= 13'd0;
            fifo_write_pointer <= {FIFO_ADDRESS_BITS{1'b0}};
            fifo_read_pointer  <= {FIFO_ADDRESS_BITS{1'b0}};
            fifo_count         <= {(FIFO_ADDRESS_BITS+1){1'b0}};
        end else begin
            if (core_output_transfer) begin
                case (pack_byte_count)
                    2'd0: pack_data[7:0]   <= core_out_pixel;
                    2'd1: pack_data[15:8]  <= core_out_pixel;
                    2'd2: pack_data[23:16] <= core_out_pixel;
                    default: pack_data[31:24] <= core_out_pixel;
                endcase

                if (pack_byte_count == 3)
                    pack_byte_count <= 2'd0;
                else
                    pack_byte_count <= pack_byte_count + 1'b1;
            end

            if (fifo_push) begin
                fifo_data[fifo_write_pointer] <= {core_out_pixel, pack_data[23:0]};
                fifo_last[fifo_write_pointer] <=
                    (pushed_word_count == WORDS_PER_PATCH - 1);
                fifo_write_pointer <= fifo_write_pointer + 1'b1;

                if (pushed_word_count == WORDS_PER_PATCH - 1)
                    pushed_word_count <= 13'd0;
                else
                    pushed_word_count <= pushed_word_count + 1'b1;
            end

            if (fifo_pop)
                fifo_read_pointer <= fifo_read_pointer + 1'b1;

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: fifo_count <= fifo_count;
            endcase
        end
    end

endmodule
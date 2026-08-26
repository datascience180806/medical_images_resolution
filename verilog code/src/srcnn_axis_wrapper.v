`timescale 1ns / 1ps

module srcnn_axis_wrapper #(
    parameter C_PIXEL_WIDTH        = 8,
    parameter C_TDATA_WIDTH        = 32,
    parameter C_S_AXIS_TDATA_WIDTH = 32,
    parameter C_M_AXIS_TDATA_WIDTH = 32,
    parameter C_FIFO_DEPTH         = 512,
    parameter C_AF_MARGIN          = 16,
    parameter FIFO_WIDTH           = 33,
    parameter FIFO_ADDR_W          = 9,
    parameter AF_THRESHOLD         = C_FIFO_DEPTH - C_AF_MARGIN
)(
    input  wire         clk,
    input  wire         rst_n,

    // AXIS Slave (MM2S DMA -> IP)
    input  wire [31:0]  s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    input  wire         s_axis_tlast,

    // AXIS Master (IP -> S2MM DMA)
    output wire [31:0]  m_axis_tdata,
    output wire         m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast
);

    wire fifo_almost_full;
    reg  [1:0]  in_byte_idx;
    reg  [31:0] in_word_hold;

    // ------------------------------------------------------------------
    // 1. Input De-serializer: 1 tu 32-bit -> 4 pixel 8-bit
    // ------------------------------------------------------------------
    assign s_axis_tready = rst_n & (in_byte_idx == 2'd0) & (~fifo_almost_full);
    wire s_xfer = s_axis_tvalid & s_axis_tready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_byte_idx  <= 2'd0;
            in_word_hold <= 32'd0;
        end else if (in_byte_idx == 2'd0) begin
            if (s_xfer) begin
                in_word_hold <= s_axis_tdata;
                in_byte_idx  <= 2'd1;
            end
        end else begin
            in_byte_idx <= (in_byte_idx == 2'd3) ? 2'd0 : (in_byte_idx + 2'd1);
        end
    end

    wire [7:0] pixel_sel = (in_byte_idx == 2'd0) ? s_axis_tdata[7:0]    :
                           (in_byte_idx == 2'd1) ? in_word_hold[15:8]  :
                           (in_byte_idx == 2'd2) ? in_word_hold[23:16] :
                                                   in_word_hold[31:24];

    wire pixel_valid = (in_byte_idx == 2'd0) ? s_xfer : 1'b1;

    // ------------------------------------------------------------------
    // 2. Loi tinh toan SRCNN 3 tang
    // ------------------------------------------------------------------
    wire [7:0] core_data_out;
    wire       core_valid_out;

    srcnn_top_core #(
        .IMG_WIDTH(128)
    ) u_srcnn_top (
        .clk        (clk),
        .rst        (~rst_n),
        .start_conv (pixel_valid),
        .pixel_in   (pixel_sel),
        .pixel_out  (core_data_out),
        .data_valid (core_valid_out)
    );

    // ------------------------------------------------------------------
    // 3. Bo gom-TU ngo ra & Pixel Dropper (Chong Deadlock)
    // ------------------------------------------------------------------
   reg  [1:0]  out_byte_idx;
    reg  [31:0] word_build;
    reg  [11:0] out_word_cnt;   // Dem 0 .. 4095 tu 32-bit (4096 x 4 = 16384 bytes)

    // Tu thu 4096 (index 4095) kich hoat TLAST cho S2MM DMA
    wire is_last_word_of_frame = (out_word_cnt == 12'd4095);

    wire [31:0] word_updated =
        (out_byte_idx == 2'd0) ? {word_build[31:8],  core_data_out} :
        (out_byte_idx == 2'd1) ? {word_build[31:16], core_data_out, word_build[7:0]} :
        (out_byte_idx == 2'd2) ? {word_build[31:24], core_data_out, word_build[15:0]} :
                                 {core_data_out, word_build[23:0]};

    wire word_complete = core_valid_out & (out_byte_idx == 2'd3);
    wire word_is_tlast  = word_complete & is_last_word_of_frame;

    wire fifo_full;
    wire fifo_wr_en     = word_complete & (~fifo_full);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_byte_idx <= 2'd0;
            word_build   <= 32'd0;
            out_word_cnt <= 12'd0;
        end else if (core_valid_out) begin
            word_build   <= word_updated;
            out_byte_idx <= (out_byte_idx == 2'd3) ? 2'd0 : (out_byte_idx + 2'd1);

            if (out_byte_idx == 2'd3) begin
                if (out_word_cnt == 12'd4095)
                    out_word_cnt <= 12'd0;
                else
                    out_word_cnt <= out_word_cnt + 1'b1;
            end
        end
    end
    // ------------------------------------------------------------------
    // 4. FIFO ngo ra Cap-Tu 32-bit
    // ------------------------------------------------------------------
    reg [FIFO_WIDTH-1:0]  fifo_mem [0:C_FIFO_DEPTH-1];
    reg [FIFO_ADDR_W-1:0] wr_ptr;
    reg [FIFO_ADDR_W-1:0] rd_ptr;
    reg [FIFO_ADDR_W:0]   fifo_count;

    assign fifo_full        = (fifo_count == C_FIFO_DEPTH);
    wire   fifo_empty       = (fifo_count == {(FIFO_ADDR_W+1){1'b0}});
    assign fifo_almost_full = (fifo_count >= AF_THRESHOLD);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= {FIFO_ADDR_W{1'b0}};
        end else if (fifo_wr_en) begin
            fifo_mem[wr_ptr] <= {word_is_tlast, word_updated};
            wr_ptr           <= wr_ptr + 1'b1;
        end
    end

    reg                   out_val_r;
    reg [FIFO_WIDTH-1:0]  out_data_r;

    wire fifo_rd_en = (~fifo_empty) & (~out_val_r | m_axis_tready);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= {FIFO_ADDR_W{1'b0}};
        end else if (fifo_rd_en) begin
            rd_ptr <= rd_ptr + 1'b1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_count <= {(FIFO_ADDR_W+1){1'b0}};
        end else begin
            case ({fifo_wr_en, fifo_rd_en})
                2'b10   : fifo_count <= fifo_count + 1'b1;
                2'b01   : fifo_count <= fifo_count - 1'b1;
                default : fifo_count <= fifo_count;
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_val_r  <= 1'b0;
            out_data_r <= {FIFO_WIDTH{1'b0}};
        end else if (fifo_rd_en) begin
            out_data_r <= fifo_mem[rd_ptr];
            out_val_r  <= 1'b1;
        end else if (m_axis_tready) begin
            out_val_r  <= 1'b0;
        end
    end

    assign m_axis_tvalid = out_val_r;
    assign m_axis_tdata  = out_data_r[31:0];
    assign m_axis_tlast  = out_val_r & out_data_r[32];

endmodule
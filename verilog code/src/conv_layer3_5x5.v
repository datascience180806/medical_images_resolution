`timescale 1ns / 1ps

module conv_layer3_5x5 #(
    parameter IMG_WIDTH = 128
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                valid_in,
    input  wire signed [15:0]  pixel_in,
    output wire [7:0]          pixel_out,
    output wire                valid_out
);

    // 1. Line Buffer 5x5
    wire signed [15:0] row0, row1, row2, row3, row4;

    line_buffer_5x5 #(
        .IMG_WIDTH(IMG_WIDTH)
    ) buf_5x5 (
        .clk       (clk),
        .rst       (rst),
        .en        (valid_in),
        .pixel_in  (pixel_in),
        .out_row0  (row0), 
        .out_row1  (row1), 
        .out_row2  (row2),
        .out_row3  (row3), 
        .out_row4  (row4)
    );

    // 2. Cua so truot 5x5
    reg signed [15:0] win [0:24];
    integer r, c;

    always @(posedge clk) begin
        if (rst) begin
            for (r = 0; r < 25; r = r + 1) win[r] <= 16'sd0;
        end else if (valid_in) begin
            for (r = 0; r < 5; r = r + 1) begin
                for (c = 4; c > 0; c = c - 1) begin
                    win[r*5 + c] <= win[r*5 + c - 1];
                end
            end
            win[0]  <= row0;
            win[5]  <= row1;
            win[10] <= row2;
            win[15] <= row3;
            win[20] <= row4;
        end
    end

    // 3. Trong so Layer 3 (Q7: 128 = 1.0)
    wire signed [15:0] w3 [0:24];
    genvar w_idx;
    generate
        for (w_idx = 0; w_idx < 25; w_idx = w_idx + 1) begin : W_INIT
            if (w_idx == 12)
                assign w3[w_idx] = 16'sd32;
            else
                assign w3[w_idx] = 16'sd4;
        end
    endgenerate

    // 4. Mang 25 bo nhan
    reg signed [31:0] mult_reg [0:24];
    integer m;

    always @(posedge clk) begin
        if (rst) begin
            for (m = 0; m < 25; m = m + 1) mult_reg[m] <= 32'sd0;
        end else begin
            for (m = 0; m < 25; m = m + 1)
                mult_reg[m] <= win[m] * w3[m];
        end
    end

    // 5. Cay cong 5 tang
    wire [25*32-1:0] mult_flat;
    genvar k;
    generate
        for (k = 0; k < 25; k = k + 1) begin : PACK
            assign mult_flat[k*32 +: 32] = mult_reg[k];
        end
    endgenerate

    wire signed [31:0] sum_tree;

    adder_tree_25 u_adder_tree_25 (
        .clk     (clk),
        .rst     (rst),
        .en      (1'b1),
        .in_p    (mult_flat),
        .sum_out (sum_tree)
    );

    // 6. Hau xu ly: Shift Q7 -> Clamp [0..255]
    localparam Q_SHIFT = 7;
    wire signed [31:0] sum_shifted = sum_tree >>> Q_SHIFT;
    wire [7:0] clamped_pixel = sum_shifted[31] ? 8'd0 :
                               (sum_shifted > 32'd255) ? 8'd255 :
                               sum_shifted[7:0];

    reg [7:0] out_reg;
    always @(posedge clk) begin
        if (rst) out_reg <= 8'd0;
        else     out_reg <= clamped_pixel;
    end

    assign pixel_out = out_reg;

    // 7. Pipeline Valid: 8 cycles
    reg [7:0] v_pipe;
    always @(posedge clk) begin
        if (rst) v_pipe <= 8'd0;
        else     v_pipe <= {v_pipe[6:0], valid_in};
    end

    assign valid_out = v_pipe[7];

endmodule
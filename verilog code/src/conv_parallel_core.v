`timescale 1ns / 1ps

module conv_parallel_core #(
    parameter IMG_WIDTH = 128
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start_conv,
    input  wire signed [15:0]   pixel_in,
    output wire signed [15:0]   pixel_out,
    output wire                 data_valid
);

    // 1. Line Buffer 9x9
    wire signed [15:0] row_out0, row_out1, row_out2, row_out3, row_out4;
    wire signed [15:0] row_out5, row_out6, row_out7, row_out8;

    line_buffer #(
        .IMG_WIDTH(IMG_WIDTH)
    ) buffer_9x9 (
        .clk       (clk),
        .rst       (rst),
        .en        (start_conv),
        .pixel_in  (pixel_in),
        .out_row0  (row_out0), .out_row1  (row_out1), .out_row2  (row_out2),
        .out_row3  (row_out3), .out_row4  (row_out4), .out_row5  (row_out5),
        .out_row6  (row_out6), .out_row7  (row_out7), .out_row8  (row_out8)
    );

    // 2. Cua so truot 9x9
    reg signed [15:0] window [0:80];
    integer r, c;

    always @(posedge clk) begin
        if (rst) begin
            for (r = 0; r < 81; r = r + 1) begin
                window[r] <= 16'sd0;
            end
        end else if (start_conv) begin
            for (r = 0; r < 9; r = r + 1) begin
                for (c = 8; c > 0; c = c - 1) begin
                    window[r*9 + c] <= window[r*9 + c - 1];
                end
            end
            window[0]  <= row_out0;
            window[9]  <= row_out1;
            window[18] <= row_out2;
            window[27] <= row_out3;
            window[36] <= row_out4;
            window[45] <= row_out5;
            window[54] <= row_out6;
            window[63] <= row_out7;
            window[72] <= row_out8;
        end
    end

    // 3. Mang 81 trong so Q7 (Gia tri tam = 64, xung quanh = 1)
    wire signed [15:0] rom_weights [0:80];
    genvar w_i;
    generate
        for (w_i = 0; w_i < 81; w_i = w_i + 1) begin : W_GEN
            if (w_i == 40) // Pixel tam giua
                assign rom_weights[w_i] = 16'sd64;
            else
                assign rom_weights[w_i] = 16'sd1;
        end
    endgenerate

    // 4. Mang 81 bo nhan song song
    reg signed [31:0] mult_reg [0:80];
    integer m;

    always @(posedge clk) begin
        if (rst) begin
            for (m = 0; m < 81; m = m + 1) mult_reg[m] <= 32'sd0;
        end else begin
            for (m = 0; m < 81; m = m + 1)
                mult_reg[m] <= window[m] * rom_weights[m];
        end
    end

    // 5. Dong goi mang 2D thanh vector 1D feed vao Adder Tree
    wire [81*32-1:0] mult_flat;
    genvar k;
    generate
        for (k = 0; k < 81; k = k + 1) begin : PACK
            assign mult_flat[k*32 +: 32] = mult_reg[k];
        end
    endgenerate

    wire signed [31:0] tree_sum;

    adder_tree_81 u_adder_tree (
        .clk     (clk),
        .rst     (rst),
        .en      (1'b1),
        .in_p    (mult_flat),
        .sum_out (tree_sum)
    );

    // 6. Hau xu ly: Shift Q7 -> Clamp [0..255]
    localparam Q_SHIFT = 7;
    wire signed [31:0] sum_shifted = tree_sum >>> Q_SHIFT;
    wire [15:0] sum_clamped = sum_shifted[31] ? 16'd0 :
                              (sum_shifted > 32'd255) ? 16'd255 :
                              sum_shifted[15:0];

    reg [15:0] pixel_out_reg;
    always @(posedge clk) begin
        if (rst) pixel_out_reg <= 16'd0;
        else     pixel_out_reg <= sum_clamped;
    end

    assign pixel_out = pixel_out_reg;

    // 7. Pipeline Valid: 10 cycles
    reg [9:0] v_pipe;
    always @(posedge clk) begin
        if (rst) v_pipe <= 10'd0;
        else     v_pipe <= {v_pipe[8:0], start_conv};
    end

    assign data_valid = v_pipe[9];

endmodule
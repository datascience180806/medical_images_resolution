`timescale 1ns / 1ps
// ============================================================================
// line_buffer.v (Layer 1 - cua so 9x9)
// Doi chieu RCA: IMG_WIDTH=128 duoc dung lam mac dinh va dinh nghia truc
// tiep kich thuoc mang RAM (0:IMG_WIDTH-1) + diem wrap con tro (IMG_WIDTH-1).
// Khong con hardcode 64 o bat ky noi nao trong file nay.
// ============================================================================

module line_buffer #(
    parameter IMG_WIDTH = 128
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                en,
    input  wire signed [15:0]  pixel_in,

    output wire signed [15:0]  out_row0,
    output wire signed [15:0]  out_row1,
    output wire signed [15:0]  out_row2,
    output wire signed [15:0]  out_row3,
    output wire signed [15:0]  out_row4,
    output wire signed [15:0]  out_row5,
    output wire signed [15:0]  out_row6,
    output wire signed [15:0]  out_row7,
    output wire signed [15:0]  out_row8
);

    reg signed [15:0] row1 [0:IMG_WIDTH-1];
    reg signed [15:0] row2 [0:IMG_WIDTH-1];
    reg signed [15:0] row3 [0:IMG_WIDTH-1];
    reg signed [15:0] row4 [0:IMG_WIDTH-1];
    reg signed [15:0] row5 [0:IMG_WIDTH-1];
    reg signed [15:0] row6 [0:IMG_WIDTH-1];
    reg signed [15:0] row7 [0:IMG_WIDTH-1];
    reg signed [15:0] row8 [0:IMG_WIDTH-1];

    reg [10:0] ptr;

    always @(posedge clk) begin
        if (rst) begin
            ptr <= 11'd0;
        end else if (en) begin
            row8[ptr] <= row7[ptr];
            row7[ptr] <= row6[ptr];
            row6[ptr] <= row5[ptr];
            row5[ptr] <= row4[ptr];
            row4[ptr] <= row3[ptr];
            row3[ptr] <= row2[ptr];
            row2[ptr] <= row1[ptr];
            row1[ptr] <= pixel_in;

            if (ptr == IMG_WIDTH - 1)
                ptr <= 11'd0;
            else
                ptr <= ptr + 1'b1;
        end
    end

    assign out_row0 = pixel_in;
    assign out_row1 = row1[ptr];
    assign out_row2 = row2[ptr];
    assign out_row3 = row3[ptr];
    assign out_row4 = row4[ptr];
    assign out_row5 = row5[ptr];
    assign out_row6 = row6[ptr];
    assign out_row7 = row7[ptr];
    assign out_row8 = row8[ptr];

endmodule

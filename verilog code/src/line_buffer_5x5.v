`timescale 1ns / 1ps

module line_buffer_5x5 #(
    parameter IMG_WIDTH = 128
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                en,
    input  wire signed [15:0]  pixel_in,

    // Xuat 5 diem anh tren cung 1 cot cho cua so 5x5
    output wire signed [15:0]  out_row0,
    output wire signed [15:0]  out_row1,
    output wire signed [15:0]  out_row2,
    output wire signed [15:0]  out_row3,
    output wire signed [15:0]  out_row4
);

    // 4 hang RAM dem cho cua so 5x5
    reg signed [15:0] row1 [0:IMG_WIDTH-1];
    reg signed [15:0] row2 [0:IMG_WIDTH-1];
    reg signed [15:0] row3 [0:IMG_WIDTH-1];
    reg signed [15:0] row4 [0:IMG_WIDTH-1];

    reg [10:0] ptr;

    // Reset dong bo: chi xoa con tro ptr, giu nguyen mang RAM
    always @(posedge clk) begin
        if (rst) begin
            ptr <= 11'd0;
        end else if (en) begin
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

endmodule
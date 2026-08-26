`timescale 1ns / 1ps

module tb_conv_core;
    reg clk, rst, start_conv;
    reg signed [15:0] pixel_in;
    wire signed [31:0] pixel_out;
    wire data_valid;

    // Khởi tạo hệ thống
    conv_core dut (
        .clk(clk), .rst(rst), .start_conv(start_conv),
        .pixel_in(pixel_in), .pixel_out(pixel_out), .data_valid(data_valid)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; start_conv = 0; pixel_in = 50;
        #20 rst = 0;
        #10 start_conv = 1; #10 start_conv = 0;
        
        $display("--- Bắt đầu mô phỏng toàn hệ thống ---");
        wait(data_valid); // Chờ cho đến khi nhân AI tính xong
        $display("Kết quả tính toán cuối cùng: %d", pixel_out);
        $finish;
    end
endmodule
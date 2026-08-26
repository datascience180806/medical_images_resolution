`timescale 1ns / 1ps

module srcnn_top_core #(
    parameter IMG_WIDTH = 128
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start_conv,
    input  wire [7:0]  pixel_in,
    output wire [7:0]  pixel_out,
    output wire        data_valid
);

    // 1. Layer 1: 9x9 Parallel Core (Giu nguyen ten cong start_conv va data_valid)
    wire signed [15:0] l1_data_out;
    wire               l1_valid_out;

    conv_parallel_core #(
        .IMG_WIDTH(IMG_WIDTH)
    ) u_layer1 (
        .clk        (clk),
        .rst        (rst),
        .start_conv (start_conv),
        .pixel_in   ({8'd0, pixel_in}),
        .pixel_out  (l1_data_out),
        .data_valid (l1_valid_out)
    );

    // 2. Layer 2: 1x1 Non-linear Mapping
    wire signed [15:0] l2_data_out;
    wire               l2_valid_out;

    conv_layer2_1x1 u_layer2 (
        .clk        (clk),
        .rst        (rst),
        .valid_in   (l1_valid_out),
        .pixel_in   (l1_data_out),
        .pixel_out  (l2_data_out),
        .valid_out  (l2_valid_out)
    );

    // 3. Layer 3: 5x5 Reconstruction
    conv_layer3_5x5 #(
        .IMG_WIDTH(IMG_WIDTH)
    ) u_layer3 (
        .clk        (clk),
        .rst        (rst),
        .valid_in   (l2_valid_out),
        .pixel_in   (l2_data_out),
        .pixel_out  (pixel_out),
        .valid_out  (data_valid)
    );

endmodule
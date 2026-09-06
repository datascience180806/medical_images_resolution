`timescale 1ns / 1ps

// Compact multi-channel SRCNN core: 1 -> 16 -> 8 -> 1.
module srcnn_top_core #(
    parameter IMAGE_WIDTH       = 128,
    parameter IMAGE_HEIGHT      = 128,
    parameter OUTPUT_ZERO_POINT = 128,
    parameter WEIGHT_FILE       = "weights_hex_clean.txt",
    parameter BIAS_FILE         = "biases_hex_clean.txt"
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire signed [7:0]    in_pixel,
    output wire                 out_valid,
    input  wire                 out_ready,
    output wire [7:0]           out_pixel
);

    wire [10367:0] l1_weights;
    wire [1023:0]  l2_weights;
    wire [1599:0]  l3_weights;
    wire [511:0]   l1_biases;
    wire [255:0]   l2_biases;
    wire [31:0]    l3_bias;

    weight_rom #(
        .WEIGHT_FILE (WEIGHT_FILE),
        .BIAS_FILE   (BIAS_FILE)
    ) coefficient_store (
        .l1_weights (l1_weights),
        .l2_weights (l2_weights),
        .l3_weights (l3_weights),
        .l1_biases  (l1_biases),
        .l2_biases  (l2_biases),
        .l3_bias    (l3_bias)
    );

    wire         l1_valid;
    wire         l1_ready;
    wire [127:0] l1_features;
    wire         l2_valid;
    wire         l2_ready;
    wire [63:0]  l2_features;

    conv_layer1_9x9 #(
        .IMAGE_WIDTH  (IMAGE_WIDTH),
        .IMAGE_HEIGHT (IMAGE_HEIGHT)
    ) layer1 (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (in_valid),
        .in_ready     (in_ready),
        .in_pixel     (in_pixel),
        .weights      (l1_weights),
        .biases       (l1_biases),
        .out_valid    (l1_valid),
        .out_ready    (l1_ready),
        .out_features (l1_features)
    );

    conv_layer2_1x1 layer2 (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (l1_valid),
        .in_ready     (l1_ready),
        .in_features  (l1_features),
        .weights      (l2_weights),
        .biases       (l2_biases),
        .out_valid    (l2_valid),
        .out_ready    (l2_ready),
        .out_features (l2_features)
    );

    conv_layer3_5x5 #(
        .IMAGE_WIDTH       (IMAGE_WIDTH),
        .IMAGE_HEIGHT      (IMAGE_HEIGHT),
        .OUTPUT_ZERO_POINT (OUTPUT_ZERO_POINT)
    ) layer3 (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (l2_valid),
        .in_ready     (l2_ready),
        .in_features  (l2_features),
        .weights      (l3_weights),
        .bias         (l3_bias),
        .out_valid    (out_valid),
        .out_ready    (out_ready),
        .out_pixel    (out_pixel)
    );

endmodule

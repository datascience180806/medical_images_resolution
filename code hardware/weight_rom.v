`timescale 1ns / 1ps

// Compact SRCNN coefficient store.
// Weight order follows PyTorch contiguous tensor order:
//   conv1 [out=16][in=1][ky=9][kx=9] : addresses    0..1295
//   conv2 [out=8 ][in=16][ky=1][kx=1]: addresses 1296..1423
//   conv3 [out=1 ][in=8][ky=5][kx=5] : addresses 1424..1623
// Bias order:
//   conv1[0..15], conv2[0..7], conv3[0]
// Biases use signed Q14 because they are added to accumulated Q14 products.
module weight_rom #(
    parameter WEIGHT_FILE = "weights_hex_clean.txt",
    parameter BIAS_FILE   = "biases_hex_clean.txt"
)(
    output wire [10367:0] l1_weights,
    output wire [ 1023:0] l2_weights,
    output wire [ 1599:0] l3_weights,
    output wire [  511:0] l1_biases,
    output wire [  255:0] l2_biases,
    output wire [   31:0] l3_bias
);

    reg [7:0]  weight_mem [0:1623];
    reg [31:0] bias_mem   [0:24];

    integer init_index;
    initial begin
        for (init_index = 0; init_index < 1624; init_index = init_index + 1)
            weight_mem[init_index] = 8'h00;
        for (init_index = 0; init_index < 25; init_index = init_index + 1)
            bias_mem[init_index] = 32'h00000000;
        $readmemh(WEIGHT_FILE, weight_mem);
        $readmemh(BIAS_FILE, bias_mem);
    end

    genvar weight_index;
    generate
        for (weight_index = 0; weight_index < 1296; weight_index = weight_index + 1) begin : GEN_L1_WEIGHT
            assign l1_weights[(weight_index*8) +: 8] = weight_mem[weight_index];
        end
        for (weight_index = 0; weight_index < 128; weight_index = weight_index + 1) begin : GEN_L2_WEIGHT
            assign l2_weights[(weight_index*8) +: 8] = weight_mem[1296 + weight_index];
        end
        for (weight_index = 0; weight_index < 200; weight_index = weight_index + 1) begin : GEN_L3_WEIGHT
            assign l3_weights[(weight_index*8) +: 8] = weight_mem[1424 + weight_index];
        end
    endgenerate

    genvar bias_index;
    generate
        for (bias_index = 0; bias_index < 16; bias_index = bias_index + 1) begin : GEN_L1_BIAS
            assign l1_biases[(bias_index*32) +: 32] = bias_mem[bias_index];
        end
        for (bias_index = 0; bias_index < 8; bias_index = bias_index + 1) begin : GEN_L2_BIAS
            assign l2_biases[(bias_index*32) +: 32] = bias_mem[16 + bias_index];
        end
    endgenerate

    assign l3_bias = bias_mem[24];

endmodule

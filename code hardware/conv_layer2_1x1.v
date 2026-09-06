`timescale 1ns / 1ps

// Layer 2: pointwise 1x1 convolution, 16 signed Q7 inputs, 8 signed Q7 outputs.
// Sixteen multipliers are shared across the eight output channels.
module conv_layer2_1x1 (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [127:0]         in_features,
    input  wire [1023:0]        weights,
    input  wire [255:0]         biases,
    output reg                  out_valid,
    input  wire                 out_ready,
    output reg  [63:0]          out_features
);

    reg [127:0] feature_register;
    reg         processing;
    reg         launching;
    reg [3:0]   launch_channel;

    assign in_ready = !processing && !out_valid;

    reg [127:0] selected_weights;
    integer select_index;
    always @* begin
        selected_weights = 128'd0;
        for (select_index = 0; select_index < 16; select_index = select_index + 1)
            selected_weights[(select_index*8) +: 8] =
                weights[(((launch_channel * 16) + select_index) * 8) +: 8];
    end

    wire               mac_out_valid;
    wire [2:0]         mac_out_channel;
    wire signed [39:0] mac_out_sum;

    srcnn_mac16_pipeline mac16 (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (launching),
        .in_tag      (launch_channel[2:0]),
        .pixels      (feature_register),
        .coefficients(selected_weights),
        .out_valid   (mac_out_valid),
        .out_tag     (mac_out_channel),
        .out_sum     (mac_out_sum)
    );

    wire signed [31:0] selected_bias;
    wire signed [39:0] bias_extended;
    wire signed [39:0] biased_sum;
    assign selected_bias = $signed(biases[(mac_out_channel*32) +: 32]);
    assign bias_extended = {{8{selected_bias[31]}}, selected_bias};
    assign biased_sum = $signed(mac_out_sum) + $signed(bias_extended);

    function [7:0] relu_q7;
        input signed [39:0] value_q14;
        reg signed [39:0] shifted;
        begin
            shifted = $signed(value_q14) >>> 7;
            if (shifted < 0)
                relu_q7 = 8'd0;
            else if (shifted > 127)
                relu_q7 = 8'd127;
            else
                relu_q7 = shifted[7:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            feature_register <= 128'd0;
            processing       <= 1'b0;
            launching        <= 1'b0;
            launch_channel   <= 4'd0;
            out_valid        <= 1'b0;
            out_features     <= 64'd0;
        end else begin
            if (out_valid && out_ready)
                out_valid <= 1'b0;

            if (in_valid && in_ready) begin
                feature_register <= in_features;
                processing       <= 1'b1;
                launching        <= 1'b1;
                launch_channel   <= 4'd0;
            end

            if (launching) begin
                if (launch_channel == 7) begin
                    launching <= 1'b0;
                end else begin
                    launch_channel <= launch_channel + 1'b1;
                end
            end

            if (mac_out_valid) begin
                out_features[(mac_out_channel*8) +: 8] <= relu_q7(biased_sum);
                if (mac_out_channel == 7) begin
                    processing <= 1'b0;
                    out_valid  <= 1'b1;
                end
            end
        end
    end

endmodule


// Pipelined reduction of 16 signed 8x8 products into one signed 40-bit sum.
// The multiplier count is 16.
module srcnn_mac16_pipeline (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [2:0]           in_tag,
    input  wire [127:0]         pixels,
    input  wire [127:0]         coefficients,
    output wire                 out_valid,
    output wire [2:0]           out_tag,
    output wire signed [39:0]   out_sum
);

    (* use_dsp = "yes" *) reg signed [39:0] product_stage [0:15];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_1 [0:7];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_2 [0:3];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_3 [0:1];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_4;
    reg [4:0] valid_pipe;
    reg [2:0] tag_pipe [0:4];

    integer index;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= 5'd0;
            sum_stage_4 <= 40'sd0;
            for (index = 0; index < 5; index = index + 1)
                tag_pipe[index] <= 3'd0;
        end else begin
            valid_pipe[0] <= in_valid;
            valid_pipe[4:1] <= valid_pipe[3:0];
            tag_pipe[0] <= in_tag;
            for (index = 1; index < 5; index = index + 1)
                tag_pipe[index] <= tag_pipe[index-1];

            for (index = 0; index < 16; index = index + 1)
                product_stage[index] <=
                    $signed(pixels[(index*8) +: 8]) *
                    $signed(coefficients[(index*8) +: 8]);
            for (index = 0; index < 8; index = index + 1)
                sum_stage_1[index] <= $signed(product_stage[index*2]) +
                                      $signed(product_stage[index*2+1]);
            for (index = 0; index < 4; index = index + 1)
                sum_stage_2[index] <= $signed(sum_stage_1[index*2]) +
                                      $signed(sum_stage_1[index*2+1]);
            for (index = 0; index < 2; index = index + 1)
                sum_stage_3[index] <= $signed(sum_stage_2[index*2]) +
                                      $signed(sum_stage_2[index*2+1]);
            sum_stage_4 <= $signed(sum_stage_3[0]) + $signed(sum_stage_3[1]);
        end
    end

    assign out_valid = valid_pipe[4];
    assign out_tag   = tag_pipe[4];
    assign out_sum   = sum_stage_4;

endmodule

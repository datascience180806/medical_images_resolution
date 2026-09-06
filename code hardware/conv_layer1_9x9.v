`timescale 1ns / 1ps

// Layer 1: 9x9 convolution, one signed Q7 input channel, 16 signed Q7 outputs.
// The datapath reuses 81 multipliers across the 16 output channels.
// Zero padding of four pixels preserves the 128x128 spatial dimensions.
module conv_layer1_9x9 #(
    parameter IMAGE_WIDTH  = 128,
    parameter IMAGE_HEIGHT = 128
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire signed [7:0]    in_pixel,
    input  wire [10367:0]       weights,
    input  wire [511:0]         biases,
    output reg                  out_valid,
    input  wire                 out_ready,
    output reg  [127:0]         out_features
);

    localparam PAD_WIDTH  = IMAGE_WIDTH + 8;
    localparam PAD_HEIGHT = IMAGE_HEIGHT + 8;

    reg signed [7:0] line_0 [0:PAD_WIDTH-1];
    reg signed [7:0] line_1 [0:PAD_WIDTH-1];
    reg signed [7:0] line_2 [0:PAD_WIDTH-1];
    reg signed [7:0] line_3 [0:PAD_WIDTH-1];
    reg signed [7:0] line_4 [0:PAD_WIDTH-1];
    reg signed [7:0] line_5 [0:PAD_WIDTH-1];
    reg signed [7:0] line_6 [0:PAD_WIDTH-1];
    reg signed [7:0] line_7 [0:PAD_WIDTH-1];
    reg signed [7:0] window [0:80];

    reg [15:0] pad_row;
    reg [15:0] pad_col;
    reg        processing;
    reg        launching;
    reg [4:0]  launch_channel;

    wire inside_source;
    wire output_position;
    wire can_advance;
    wire step_fire;
    wire signed [7:0] step_pixel;

    assign inside_source = (pad_row >= 4) && (pad_row < IMAGE_HEIGHT + 4) &&
                           (pad_col >= 4) && (pad_col < IMAGE_WIDTH  + 4);
    assign output_position = (pad_row >= 8) && (pad_col >= 8);
    assign can_advance = !processing && !out_valid;
    assign in_ready = can_advance && inside_source;
    assign step_fire = can_advance && (!inside_source || in_valid);
    assign step_pixel = inside_source ? in_pixel : 8'sd0;

    wire [647:0] window_bus;
    genvar window_index;
    generate
        for (window_index = 0; window_index < 81; window_index = window_index + 1) begin : GEN_WINDOW_BUS
            assign window_bus[(window_index*8) +: 8] = window[window_index];
        end
    endgenerate

    reg [647:0] selected_weights;
    integer select_index;
    always @* begin
        selected_weights = 648'd0;
        for (select_index = 0; select_index < 81; select_index = select_index + 1)
            selected_weights[(select_index*8) +: 8] =
                weights[(((launch_channel * 81) + select_index) * 8) +: 8];
    end

    wire               mac_out_valid;
    wire [3:0]         mac_out_channel;
    wire signed [39:0] mac_out_sum;

    srcnn_mac81_pipeline mac81 (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (launching),
        .in_tag     (launch_channel[3:0]),
        .pixels     (window_bus),
        .coefficients(selected_weights),
        .out_valid  (mac_out_valid),
        .out_tag    (mac_out_channel),
        .out_sum    (mac_out_sum)
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

    integer row_index;
    integer column_index;
    integer reset_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            pad_row        <= 16'd0;
            pad_col        <= 16'd0;
            processing     <= 1'b0;
            launching      <= 1'b0;
            launch_channel <= 5'd0;
            out_valid      <= 1'b0;
            out_features   <= 128'd0;
            for (reset_index = 0; reset_index < 81; reset_index = reset_index + 1)
                window[reset_index] <= 8'sd0;
        end else begin
            if (out_valid && out_ready)
                out_valid <= 1'b0;

            if (step_fire) begin
                for (row_index = 0; row_index < 9; row_index = row_index + 1)
                    for (column_index = 0; column_index < 8; column_index = column_index + 1)
                        window[(row_index*9) + column_index] <=
                            window[(row_index*9) + column_index + 1];

                window[8]  <= line_7[pad_col];
                window[17] <= line_6[pad_col];
                window[26] <= line_5[pad_col];
                window[35] <= line_4[pad_col];
                window[44] <= line_3[pad_col];
                window[53] <= line_2[pad_col];
                window[62] <= line_1[pad_col];
                window[71] <= line_0[pad_col];
                window[80] <= step_pixel;

                line_7[pad_col] <= line_6[pad_col];
                line_6[pad_col] <= line_5[pad_col];
                line_5[pad_col] <= line_4[pad_col];
                line_4[pad_col] <= line_3[pad_col];
                line_3[pad_col] <= line_2[pad_col];
                line_2[pad_col] <= line_1[pad_col];
                line_1[pad_col] <= line_0[pad_col];
                line_0[pad_col] <= step_pixel;

                if (pad_col == PAD_WIDTH - 1) begin
                    pad_col <= 16'd0;
                    if (pad_row == PAD_HEIGHT - 1)
                        pad_row <= 16'd0;
                    else
                        pad_row <= pad_row + 1'b1;
                end else begin
                    pad_col <= pad_col + 1'b1;
                end

                if (output_position) begin
                    processing     <= 1'b1;
                    launching      <= 1'b1;
                    launch_channel <= 5'd0;
                end
            end

            if (launching) begin
                if (launch_channel == 15) begin
                    launching <= 1'b0;
                end else begin
                    launch_channel <= launch_channel + 1'b1;
                end
            end

            if (mac_out_valid) begin
                out_features[(mac_out_channel*8) +: 8] <= relu_q7(biased_sum);
                if (mac_out_channel == 15) begin
                    processing <= 1'b0;
                    out_valid  <= 1'b1;
                end
            end
        end
    end

endmodule


// Fully pipelined 81-product reduction tree. All internal sums are signed 40-bit.
// One channel can enter each clock. The multiplier count is 81.
module srcnn_mac81_pipeline (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [3:0]           in_tag,
    input  wire [647:0]         pixels,
    input  wire [647:0]         coefficients,
    output wire                 out_valid,
    output wire [3:0]           out_tag,
    output wire signed [39:0]   out_sum
);

    (* use_dsp = "yes" *) reg signed [39:0] product_stage [0:80];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_1 [0:40];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_2 [0:20];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_3 [0:10];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_4 [0:5];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_5 [0:2];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_6 [0:1];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_7;
    reg [7:0] valid_pipe;
    reg [3:0] tag_pipe [0:7];

    integer index;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= 8'd0;
            sum_stage_7 <= 40'sd0;
            for (index = 0; index < 8; index = index + 1)
                tag_pipe[index] <= 4'd0;
        end else begin
            valid_pipe[0] <= in_valid;
            valid_pipe[7:1] <= valid_pipe[6:0];
            tag_pipe[0] <= in_tag;
            for (index = 1; index < 8; index = index + 1)
                tag_pipe[index] <= tag_pipe[index-1];

            for (index = 0; index < 81; index = index + 1)
                product_stage[index] <=
                    $signed(pixels[(index*8) +: 8]) *
                    $signed(coefficients[(index*8) +: 8]);

            for (index = 0; index < 40; index = index + 1)
                sum_stage_1[index] <= $signed(product_stage[index*2]) +
                                      $signed(product_stage[index*2+1]);
            sum_stage_1[40] <= product_stage[80];

            for (index = 0; index < 20; index = index + 1)
                sum_stage_2[index] <= $signed(sum_stage_1[index*2]) +
                                      $signed(sum_stage_1[index*2+1]);
            sum_stage_2[20] <= sum_stage_1[40];

            for (index = 0; index < 10; index = index + 1)
                sum_stage_3[index] <= $signed(sum_stage_2[index*2]) +
                                      $signed(sum_stage_2[index*2+1]);
            sum_stage_3[10] <= sum_stage_2[20];

            for (index = 0; index < 5; index = index + 1)
                sum_stage_4[index] <= $signed(sum_stage_3[index*2]) +
                                      $signed(sum_stage_3[index*2+1]);
            sum_stage_4[5] <= sum_stage_3[10];

            for (index = 0; index < 3; index = index + 1)
                sum_stage_5[index] <= $signed(sum_stage_4[index*2]) +
                                      $signed(sum_stage_4[index*2+1]);

            sum_stage_6[0] <= $signed(sum_stage_5[0]) + $signed(sum_stage_5[1]);
            sum_stage_6[1] <= sum_stage_5[2];
            sum_stage_7 <= $signed(sum_stage_6[0]) + $signed(sum_stage_6[1]);
        end
    end

    assign out_valid = valid_pipe[7];
    assign out_tag   = tag_pipe[7];
    assign out_sum   = sum_stage_7;

endmodule

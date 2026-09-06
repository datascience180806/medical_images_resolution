`timescale 1ns / 1ps

// Layer 3: 5x5 convolution, eight signed Q7 input channels, one output channel.
// Twenty-five multipliers are reused across the eight input channels.
// The output zero point converts the centered network output back to uint8 pixels.
module conv_layer3_5x5 #(
    parameter IMAGE_WIDTH      = 128,
    parameter IMAGE_HEIGHT     = 128,
    parameter OUTPUT_ZERO_POINT = 128
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    output wire                 in_ready,
    input  wire [63:0]          in_features,
    input  wire [1599:0]        weights,
    input  wire [31:0]          bias,
    output reg                  out_valid,
    input  wire                 out_ready,
    output reg  [7:0]           out_pixel
);

    localparam PAD_WIDTH  = IMAGE_WIDTH + 4;
    localparam PAD_HEIGHT = IMAGE_HEIGHT + 4;

    reg [63:0] line_0 [0:PAD_WIDTH-1];
    reg [63:0] line_1 [0:PAD_WIDTH-1];
    reg [63:0] line_2 [0:PAD_WIDTH-1];
    reg [63:0] line_3 [0:PAD_WIDTH-1];
    reg [63:0] window [0:24];

    reg [15:0] pad_row;
    reg [15:0] pad_col;
    reg        processing;
    reg        launching;
    reg [3:0]  launch_channel;
    reg signed [39:0] channel_accumulator;

    wire inside_source;
    wire output_position;
    wire can_advance;
    wire step_fire;
    wire [63:0] step_features;

    assign inside_source = (pad_row >= 2) && (pad_row < IMAGE_HEIGHT + 2) &&
                           (pad_col >= 2) && (pad_col < IMAGE_WIDTH  + 2);
    assign output_position = (pad_row >= 4) && (pad_col >= 4);
    assign can_advance = !processing && !out_valid;
    assign in_ready = can_advance && inside_source;
    assign step_fire = can_advance && (!inside_source || in_valid);
    assign step_features = inside_source ? in_features : 64'd0;

    reg [199:0] selected_pixels;
    reg [199:0] selected_weights;
    integer select_index;
    always @* begin
        selected_pixels  = 200'd0;
        selected_weights = 200'd0;
        for (select_index = 0; select_index < 25; select_index = select_index + 1) begin
            selected_pixels[(select_index*8) +: 8] =
                window[select_index][(launch_channel*8) +: 8];
            selected_weights[(select_index*8) +: 8] =
                weights[(((launch_channel*25) + select_index)*8) +: 8];
        end
    end

    wire               mac_out_valid;
    wire [2:0]         mac_out_channel;
    wire signed [39:0] mac_out_sum;

    srcnn_mac25_pipeline mac25 (
        .clk          (clk),
        .rst_n        (rst_n),
        .in_valid     (launching),
        .in_tag       (launch_channel[2:0]),
        .pixels       (selected_pixels),
        .coefficients (selected_weights),
        .out_valid    (mac_out_valid),
        .out_tag      (mac_out_channel),
        .out_sum      (mac_out_sum)
    );

    wire signed [31:0] bias_signed;
    wire signed [39:0] bias_extended;
    wire signed [39:0] final_accumulator;
    assign bias_signed = $signed(bias);
    assign bias_extended = {{8{bias_signed[31]}}, bias_signed};
    assign final_accumulator = $signed(channel_accumulator) +
                               $signed(mac_out_sum) + $signed(bias_extended);

    function [7:0] clamp_uint8;
        input signed [39:0] value_q14;
        reg signed [39:0] shifted;
        reg signed [40:0] adjusted;
        begin
            shifted = $signed(value_q14) >>> 7;
            adjusted = $signed(shifted) + OUTPUT_ZERO_POINT;
            if (adjusted < 0)
                clamp_uint8 = 8'd0;
            else if (adjusted > 255)
                clamp_uint8 = 8'd255;
            else
                clamp_uint8 = adjusted[7:0];
        end
    endfunction

    integer row_index;
    integer column_index;
    integer reset_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            pad_row             <= 16'd0;
            pad_col             <= 16'd0;
            processing          <= 1'b0;
            launching           <= 1'b0;
            launch_channel      <= 4'd0;
            channel_accumulator <= 40'sd0;
            out_valid           <= 1'b0;
            out_pixel           <= 8'd0;
            for (reset_index = 0; reset_index < 25; reset_index = reset_index + 1)
                window[reset_index] <= 64'd0;
        end else begin
            if (out_valid && out_ready)
                out_valid <= 1'b0;

            if (step_fire) begin
                for (row_index = 0; row_index < 5; row_index = row_index + 1)
                    for (column_index = 0; column_index < 4; column_index = column_index + 1)
                        window[(row_index*5) + column_index] <=
                            window[(row_index*5) + column_index + 1];

                window[4]  <= line_3[pad_col];
                window[9]  <= line_2[pad_col];
                window[14] <= line_1[pad_col];
                window[19] <= line_0[pad_col];
                window[24] <= step_features;

                line_3[pad_col] <= line_2[pad_col];
                line_2[pad_col] <= line_1[pad_col];
                line_1[pad_col] <= line_0[pad_col];
                line_0[pad_col] <= step_features;

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
                    processing          <= 1'b1;
                    launching           <= 1'b1;
                    launch_channel      <= 4'd0;
                    channel_accumulator <= 40'sd0;
                end
            end

            if (launching) begin
                if (launch_channel == 7) begin
                    launching <= 1'b0;
                end else begin
                    launch_channel <= launch_channel + 1'b1;
                end
            end

            if (mac_out_valid) begin
                if (mac_out_channel == 0)
                    channel_accumulator <= mac_out_sum;
                else
                    channel_accumulator <= $signed(channel_accumulator) + $signed(mac_out_sum);

                if (mac_out_channel == 7) begin
                    out_pixel  <= clamp_uint8(final_accumulator);
                    out_valid  <= 1'b1;
                    processing <= 1'b0;
                end
            end
        end
    end

endmodule


// Pipelined reduction of 25 signed 8x8 products into one signed 40-bit sum.
// The multiplier count is 25.
module srcnn_mac25_pipeline (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire [2:0]           in_tag,
    input  wire [199:0]         pixels,
    input  wire [199:0]         coefficients,
    output wire                 out_valid,
    output wire [2:0]           out_tag,
    output wire signed [39:0]   out_sum
);

    (* use_dsp = "yes" *) reg signed [39:0] product_stage [0:24];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_1 [0:12];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_2 [0:6];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_3 [0:3];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_4 [0:1];
    (* use_dsp = "no" *) reg signed [39:0] sum_stage_5;
    reg [5:0] valid_pipe;
    reg [2:0] tag_pipe [0:5];

    integer index;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= 6'd0;
            sum_stage_5 <= 40'sd0;
            for (index = 0; index < 6; index = index + 1)
                tag_pipe[index] <= 3'd0;
        end else begin
            valid_pipe[0] <= in_valid;
            valid_pipe[5:1] <= valid_pipe[4:0];
            tag_pipe[0] <= in_tag;
            for (index = 1; index < 6; index = index + 1)
                tag_pipe[index] <= tag_pipe[index-1];

            for (index = 0; index < 25; index = index + 1)
                product_stage[index] <=
                    $signed(pixels[(index*8) +: 8]) *
                    $signed(coefficients[(index*8) +: 8]);

            for (index = 0; index < 12; index = index + 1)
                sum_stage_1[index] <= $signed(product_stage[index*2]) +
                                      $signed(product_stage[index*2+1]);
            sum_stage_1[12] <= product_stage[24];

            for (index = 0; index < 6; index = index + 1)
                sum_stage_2[index] <= $signed(sum_stage_1[index*2]) +
                                      $signed(sum_stage_1[index*2+1]);
            sum_stage_2[6] <= sum_stage_1[12];

            for (index = 0; index < 3; index = index + 1)
                sum_stage_3[index] <= $signed(sum_stage_2[index*2]) +
                                      $signed(sum_stage_2[index*2+1]);
            sum_stage_3[3] <= sum_stage_2[6];

            sum_stage_4[0] <= $signed(sum_stage_3[0]) + $signed(sum_stage_3[1]);
            sum_stage_4[1] <= $signed(sum_stage_3[2]) + $signed(sum_stage_3[3]);
            sum_stage_5 <= $signed(sum_stage_4[0]) + $signed(sum_stage_4[1]);
        end
    end

    assign out_valid = valid_pipe[5];
    assign out_tag   = tag_pipe[5];
    assign out_sum   = sum_stage_5;

endmodule

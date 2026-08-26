`timescale 1ns / 1ps

module conv_layer2_1x1 (
    input  wire                clk,
    input  wire                rst,
    input  wire                valid_in,
    input  wire signed [15:0]  pixel_in,   // Ngo vao Q7 tu Layer 1
    output wire signed [15:0]  pixel_out,  // Ngo ra Q7 feed sang Layer 3
    output wire                valid_out
);

    // Trong so va Bias mac dinh cho Layer 2 (Q7 format)
    localparam signed [15:0] W2_Q7   = 16'sd128; // Trong so mau 1.0 (Q7)
    localparam signed [31:0] BIAS_Q7 = 32'sd0;
    localparam Q_SHIFT = 7;

    // Stage 1: Phep nhan 1x1
    reg signed [31:0] mult_r;
    always @(posedge clk) begin
        if (rst) mult_r <= 32'sd0;
        else     mult_r <= pixel_in * W2_Q7;
    end

    // Stage 2: Cong Bias + Dich bit Q7
    reg signed [31:0] sum_r;
    always @(posedge clk) begin
        if (rst) sum_r <= 32'sd0;
        else     sum_r <= (mult_r + BIAS_Q7) >>> Q_SHIFT;
    end

    // Stage 3: Ham kich hoat ReLU: max(0, x)
    reg signed [15:0] relu_r;
    always @(posedge clk) begin
        if (rst) 
            relu_r <= 16'sd0;
        else if (sum_r[31]) 
            relu_r <= 16'sd0; // So am -> cat ve 0
        else if (sum_r > 32'sd32767) 
            relu_r <= 16'sd32767;
        else 
            relu_r <= sum_r[15:0];
    end

    assign pixel_out = relu_r;

    // Pipeline dong bo Valid (3 cycles)
    reg [2:0] v_pipe;
    always @(posedge clk) begin
        if (rst) v_pipe <= 3'b0;
        else     v_pipe <= {v_pipe[1:0], valid_in};
    end

    assign valid_out = v_pipe[2];

endmodule
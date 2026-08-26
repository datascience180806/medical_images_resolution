`timescale 1ns / 1ps

module adder_tree_25 (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      en,
    input  wire signed [25*32-1:0]   in_p,
    output wire signed [31:0]        sum_out
);

    // Giai nen vector 1D ve mang 2D
    wire signed [31:0] in_arr [0:24];
    genvar g;
    generate
        for (g = 0; g < 25; g = g + 1) begin : UNPACK
            assign in_arr[g] = in_p[g*32 +: 32];
        end
    endgenerate

    integer i;

    // Stage 1: 25 -> 13 (12 cap + 1 phan tu le)
    reg signed [31:0] s1 [0:12];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 12; i = i + 1) s1[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 12; i = i + 1)
                s1[i] <= in_arr[2*i] + in_arr[2*i + 1];
            s1[12] <= in_arr[24];
        end
    end

    // Stage 2: 13 -> 7 (6 cap + 1 phan tu le)
    reg signed [31:0] s2 [0:6];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 6; i = i + 1) s2[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 6; i = i + 1)
                s2[i] <= s1[2*i] + s1[2*i + 1];
            s2[6] <= s1[12];
        end
    end

    // Stage 3: 7 -> 4 (3 cap + 1 phan tu le)
    reg signed [31:0] s3 [0:3];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 3; i = i + 1) s3[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 3; i = i + 1)
                s3[i] <= s2[2*i] + s2[2*i + 1];
            s3[3] <= s2[6];
        end
    end

    // Stage 4: 4 -> 2 (2 cap)
    reg signed [31:0] s4 [0:1];
    always @(posedge clk) begin
        if (rst) begin
            s4[0] <= 32'sd0;
            s4[1] <= 32'sd0;
        end else if (en) begin
            s4[0] <= s3[0] + s3[1];
            s4[1] <= s3[2] + s3[3];
        end
    end

    // Stage 5: 2 -> 1
    reg signed [31:0] s5;
    always @(posedge clk) begin
        if (rst) begin
            s5 <= 32'sd0;
        end else if (en) begin
            s5 <= s4[0] + s4[1];
        end
    end

    assign sum_out = s5;

endmodule
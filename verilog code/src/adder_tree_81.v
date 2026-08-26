`timescale 1ns / 1ps

module adder_tree_81 (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      en,
    input  wire signed [81*32-1:0]   in_p,   // Trai phang 81 phan tu 32-bit (2592-bit)
    output wire signed [31:0]        sum_out
);

    // 1. Giai nen vector 1D ve mang 2D noi bo
    wire signed [31:0] in_arr [0:80];
    genvar g;
    generate
        for (g = 0; g < 81; g = g + 1) begin : UNPACK
            assign in_arr[g] = in_p[g*32 +: 32];
        end
    endgenerate

    integer i;

    // Stage 1: 81 -> 41
    reg signed [31:0] s1 [0:40];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 40; i = i + 1) s1[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 40; i = i + 1)
                s1[i] <= in_arr[2*i] + in_arr[2*i + 1];
            s1[40] <= in_arr[80]; // Phan tu le
        end
    end

    // Stage 2: 41 -> 21
    reg signed [31:0] s2 [0:20];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 20; i = i + 1) s2[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 20; i = i + 1)
                s2[i] <= s1[2*i] + s1[2*i + 1];
            s2[20] <= s1[40];
        end
    end

    // Stage 3: 21 -> 11
    reg signed [31:0] s3 [0:10];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 10; i = i + 1) s3[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 10; i = i + 1)
                s3[i] <= s2[2*i] + s2[2*i + 1];
            s3[10] <= s2[20];
        end
    end

    // Stage 4: 11 -> 6
    reg signed [31:0] s4 [0:5];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 5; i = i + 1) s4[i] <= 32'sd0;
        end else if (en) begin
            for (i = 0; i < 5; i = i + 1)
                s4[i] <= s3[2*i] + s3[2*i + 1];
            s4[5] <= s3[10];
        end
    end

    // Stage 5: 6 -> 3
    reg signed [31:0] s5 [0:2];
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= 2; i = i + 1) s5[i] <= 32'sd0;
        end else if (en) begin
            s5[0] <= s4[0] + s4[1];
            s5[1] <= s4[2] + s4[3];
            s5[2] <= s4[4] + s4[5];
        end
    end

    // Stage 6: 3 -> 2
    reg signed [31:0] s6 [0:1];
    always @(posedge clk) begin
        if (rst) begin
            s6[0] <= 32'sd0;
            s6[1] <= 32'sd0;
        end else if (en) begin
            s6[0] <= s5[0] + s5[1];
            s6[1] <= s5[2];
        end
    end

    // Stage 7: 2 -> 1
    reg signed [31:0] s7;
    always @(posedge clk) begin
        if (rst) begin
            s7 <= 32'sd0;
        end else if (en) begin
            s7 <= s6[0] + s6[1];
        end
    end

    assign sum_out = s7;

endmodule
`timescale 1ns / 1ps

module mac_unit (
    input  wire                clk,
    input  wire                rst,        // Reset he thong toan cuc (active-high)
    input  wire                clr,        // mac_clr tu FSM: xoa accumulator tai nhip dau
    input  wire                en,         // mac_en tu FSM: bat bo tinh
    input  wire signed [15:0]  pixel_in,
    input  wire signed [15:0]  weight_in,
    output wire signed [31:0]  mac_out
);

    // ==========================================
    // TANG 1: THANH GHI NGO VAO (A-REG / B-REG)
    // ==========================================
    reg signed [15:0] pixel_in_r;
    reg signed [15:0] weight_in_r;
    reg                en_stage1;
    reg                clr_stage1;

    always @(posedge clk) begin
        if (rst) begin
            pixel_in_r  <= 16'sd0;
            weight_in_r <= 16'sd0;
            en_stage1   <= 1'b0;
            clr_stage1  <= 1'b0;
        end else begin
            pixel_in_r  <= pixel_in;
            weight_in_r <= weight_in;
            en_stage1   <= en;
            clr_stage1  <= clr;
        end
    end

    // ==========================================
    // PHEP NHAN (DSP Multiplier)
    // ==========================================
    wire signed [31:0] mult_result = pixel_in_r * weight_in_r;

    // ==========================================
    // TANG 2: THANH GHI NGO RA BO NHAN (MREG)
    // ==========================================
    reg signed [31:0] mult_out_reg;
    reg                en_stage2;
    reg                clr_stage2;

    always @(posedge clk) begin
        if (rst) begin
            mult_out_reg <= 32'sd0;
            en_stage2    <= 1'b0;
            clr_stage2   <= 1'b0;
        end else begin
            mult_out_reg <= mult_result;
            en_stage2    <= en_stage1;
            clr_stage2   <= clr_stage1;
        end
    end

    // ==========================================
    // TANG 3: ACCUMULATOR (PREG / ALU)
    // ==========================================
    reg signed [31:0] acc_reg;

    always @(posedge clk) begin
        if (rst) begin
            acc_reg <= 32'sd0;
        end else if (clr_stage2) begin
            // Nap truc tiep tich dau tien -> tu dong xoa gia tri cua diem anh truoc
            acc_reg <= mult_out_reg;
        end else if (en_stage2) begin
            // Cong don 80 tich tiep theo
            acc_reg <= acc_reg + mult_out_reg;
        end
        // Neu khong en, acc_reg GIU NGUYEN gia tri on dinh cho wrapper doc
    end

    assign mac_out = acc_reg;

endmodule
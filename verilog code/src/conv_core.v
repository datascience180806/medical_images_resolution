`timescale 1ns / 1ps

module conv_core (
    input wire clk,
    input wire rst,
    input wire start_conv,               
    input wire signed [15:0] pixel_in,   
    output wire signed [31:0] pixel_out, 
    output wire data_valid,               
    output wire core_ready                 
);

    wire mac_en;
    wire mac_clr;
    wire [6:0] weight_addr;         
    wire signed [15:0] weight_data;

    wire signed [15:0] row_out0, row_out1, row_out2, row_out3, row_out4;
    wire signed [15:0] row_out5, row_out6, row_out7, row_out8;

    wire pixel_load_pulse = start_conv & core_ready;

    line_buffer buffer_9x9 (
        .clk(clk),
        .rst(rst),
        .en(pixel_load_pulse),  
        .pixel_in(pixel_in),
        .out_row0(row_out0), .out_row1(row_out1), .out_row2(row_out2),
        .out_row3(row_out3), .out_row4(row_out4), .out_row5(row_out5),
        .out_row6(row_out6), .out_row7(row_out7), .out_row8(row_out8)
    );

    reg signed [15:0] window [0:80];
    integer r, c;

    always @(posedge clk) begin
        if (rst) begin
            for (r = 0; r < 81; r = r + 1) begin
                window[r] <= 16'd0;
            end
        end else if (pixel_load_pulse) begin
            for (r = 0; r < 9; r = r + 1) begin
                for (c = 8; c > 0; c = c - 1) begin
                    window[r*9 + c] <= window[r*9 + c - 1];
                end
            end
            window[0]  <= row_out0;
            window[9]  <= row_out1;
            window[18] <= row_out2;
            window[27] <= row_out3;
            window[36] <= row_out4;
            window[45] <= row_out5;
            window[54] <= row_out6;
            window[63] <= row_out7;
            window[72] <= row_out8;
        end
    end

    wire signed [15:0] selected_pixel = window[weight_addr];

    wire data_valid_raw;

    conv_fsm brain (
        .clk(clk),
        .rst(rst),
        .start_conv(start_conv),
        .mac_en(mac_en),
        .mac_clr(mac_clr),
        .weight_addr(weight_addr), 
        .data_valid(data_valid_raw),
        .core_ready(core_ready)     
    );

    // Can dong bo data_valid voi 2 tang pipeline cua mac_unit
    reg data_valid_d1, data_valid_d2;
    always @(posedge clk) begin
        if (rst) begin
            data_valid_d1 <= 1'b0;
            data_valid_d2 <= 1'b0;
        end else begin
            data_valid_d1 <= data_valid_raw;
            data_valid_d2 <= data_valid_d1;
        end
    end

    assign data_valid = data_valid_d2;

    weight_rom rom (
        .clk(clk),
        .addr(weight_addr),
        .weight_out(weight_data)
    );

    mac_unit heart (
        .clk(clk),
        .rst(rst),
        .clr(mac_clr),
        .en(mac_en),
        .pixel_in(selected_pixel),   
        .weight_in(weight_data),
        .mac_out(pixel_out)
    );

endmodule
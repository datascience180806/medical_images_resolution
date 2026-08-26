`timescale 1ns / 1ps

module conv_fsm (
    input wire clk,
    input wire rst,
    input wire start_conv,        
    output reg mac_en,            
    output reg mac_clr,           // Xung xoa nap accumulator tai nhip dau tien
    output reg [6:0] weight_addr, 
    output reg data_valid,        
    output wire core_ready        
);

    parameter IDLE = 2'b00; 
    parameter CALC = 2'b01; 
    parameter DONE = 2'b10; 

    reg [1:0] state;
    reg [6:0] count;        

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state       <= IDLE;
            count       <= 7'd0;
            mac_en      <= 1'b0;
            mac_clr     <= 1'b0;
            data_valid  <= 1'b0;
            weight_addr <= 7'd0;
        end else begin
            case (state)
                IDLE: begin
                    data_valid <= 1'b0;
                    mac_clr    <= 1'b0;
                    if (start_conv) begin
                        state       <= CALC;        
                        mac_en      <= 1'b1;
                        mac_clr     <= 1'b1; // Phat xung clr tai nhip count = 0
                        count       <= 7'd0;
                        weight_addr <= 7'd0;
                    end else begin
                        mac_en <= 1'b0;
                    end
                end

                CALC: begin
                    mac_clr <= 1'b0; // Ha xung clr ngay sau nhip dau
                    if (count == 7'd80) begin
                        state  <= DONE;        
                        mac_en <= 1'b0;        
                    end else begin
                        count       <= count + 1'b1;           
                        weight_addr <= count + 1'b1;     
                    end
                end

                DONE: begin
                    data_valid <= 1'b1;          
                    state      <= IDLE;          
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    assign core_ready = (state == IDLE);

endmodule
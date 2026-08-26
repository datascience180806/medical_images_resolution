`timescale 1ns / 1ps

module weight_rom (
    input wire clk,
    input wire [6:0] addr,               
    output wire signed [15:0] weight_out  
);

    // Khai báo mảng ROM: 81 phần tử, mỗi phần tử 16-bit
    reg signed [15:0] memory [0:80];

    integer i;
    integer fd_check;   

   initial begin
        // Quét sạch mảng chống rác 'X'
        for (i = 0; i <= 80; i = i + 1) begin
            memory[i] = 16'h0000;
        end
        
        // Dùng đường dẫn tuyệt đối y chang Testbench ảnh!
        $readmemh("C:/SRCNN_PS_DMA_Vivado_Project/2 file bit and hwh/dma_txt_buffers/weights_hex_clean.txt", memory, 0, 80);
        
        // Kiểm tra đúng cái đường dẫn đó
        fd_check = $fopen("C:/SRCNN_PS_DMA_Vivado_Project/2 file bit and hwh/dma_txt_buffers/weights_hex_clean.txt", "r");
        
        if (fd_check == 0) begin
            $display("=========================================================================");
            $display("[LOI NGHIEM TRONG] KHONG DOC DUOC FILE ROM!");
            $display("=========================================================================");
        end else begin
            $fclose(fd_check);
            $display("=========================================================================");
            $display("[THÀNH CÔNG] Đã nạp 81 dòng trọng số từ ổ C!");
            $display("=========================================================================");
        end
    end

    // Xuất trọng số ra theo địa chỉ (Tức thời để không bị lệch nhịp với mảng pixel)
    assign weight_out = memory[addr];

endmodule
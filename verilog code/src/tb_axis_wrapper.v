`timescale 1ns / 1ps

module tb_axis_wrapper();

    // =======================================================
    // 1. KHAI BÁO TÍN HI?U ?I?U KHI?N (Clock & Reset)
    // =======================================================
    reg clk;
    reg rst_n;

    // =======================================================
    // 2. GIAO TI?P AXI-STREAM NGÕ VÀO & NGÕ RA
    // =======================================================
    reg [7:0]  s_axis_tdata;  
    reg        s_axis_tvalid; 
    reg        s_axis_tlast;  
    wire       s_axis_tready; 

    wire [7:0] m_axis_tdata;  
    wire       m_axis_tvalid; 
    wire       m_axis_tlast;  
    reg        m_axis_tready; 

    // =======================================================
    // 3. G?I LÕI THI?T K? (DUT)
    // =======================================================
    srcnn_axis_wrapper DUT (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tready(m_axis_tready)
    );

    // =======================================================
    // 4. T?O XUNG CLOCK (10ns -> 100MHz)
    // =======================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // =======================================================
    // 5. KHAI BÁO BI?N CHO TESTBENCH & B? ?O ??NH TH?I
    // =======================================================
    integer file_id;
    integer scan_return;
    integer temp_value;
    integer pixel_count;
    
    integer file_out_id;
    integer out_pixel_count;

    // Bi?n ?o th?i gian & chu k?
    time start_time;
    time end_time;
    real elapsed_time_ns;
    real patch_time_ms;
    real frame_time_ms;
    real fps_val;
    integer total_cycles;
    reg timer_started;

    // =======================================================
    // 6. LU?NG 1: B?M D? LI?U VÀO (HOST -> LÕI AI)
    // =======================================================
    initial begin
        // Kh?i t?o tr?ng thái an toàn
        rst_n = 0; s_axis_tdata = 0; s_axis_tvalid = 0; s_axis_tlast = 0; m_axis_tready = 1;
        timer_started = 0;
        #40; 
        rst_n = 1; 
        #20;
        
        // --- N?P ?NH TILE 0005 T? ?ÚNG ???NG D?N C?A ÔNG ---
        file_id = $fopen("C:/SRCNN_PS_DMA_Vivado_Project/2 file bit and hwh/sample_lr_input.txt", "r");
        if (file_id == 0) begin 
            $display("[L?I] Không tìm th?y file ?nh test!"); 
            $finish; 
        end

        $display("[MÔ PH?NG] B?t ??u b?m m?ng ?nh tile_0005...");
        pixel_count = 0;
        scan_return = 1;
        
        while (!$feof(file_id) && scan_return == 1) begin
            scan_return = $fscanf(file_id, "%x  ", temp_value);
            
            if (scan_return == 1) begin
                s_axis_tvalid = 1'b1;
                s_axis_tdata  = temp_value[7:0];
                pixel_count   = pixel_count + 1;

                if (pixel_count == 4096) s_axis_tlast = 1'b1;
                else s_axis_tlast = 1'b0;

                // Ghi nh?n th?i ?i?m b?t ??u truy?n pixel ??u tiên (Handshake thành công)
                if (!timer_started) begin
                    start_time = $time;
                    timer_started = 1;
                end

                @(posedge clk);
                while (s_axis_tready == 1'b0) @(posedge clk);
            end
        end
        
        s_axis_tvalid = 1'b0; s_axis_tlast = 1'b0; 
        $fclose(file_id);
        $display("[MÔ PH?NG] B?m ?nh thành công! ?ang ch? m?ch tính toán...");
        #50000; 
        $display("[C?NH BÁO] Quá th?i gian ch? (Timeout)! Ng?t mô ph?ng.");
        $finish;
    end

    // =======================================================
    // 7. LU?NG 2: H?NG D? LI?U RA & XU?T BÁO CÁO HI?U N?NG
    // =======================================================
    initial begin
        file_out_id = $fopen("C:/SRCNN_PS_DMA_Vivado_Project/2 file bit and hwh/output_fpga.txt", "w");
        out_pixel_count = 0;

        if (file_out_id == 0) begin 
            $display("[L?I] Không th? t?o file output_fpga.txt!"); 
            $finish; 
        end

        forever begin
            @(posedge clk);
            if (m_axis_tvalid == 1'b1 && m_axis_tready == 1'b1) begin
                
                $fdisplay(file_out_id, "%d", m_axis_tdata);
                out_pixel_count = out_pixel_count + 1;

                if (out_pixel_count % 1000 == 0) begin
                    $display("[MÔ PH?NG] ?ã xu?t ???c %d pixel ngõ ra...", out_pixel_count);
                end

                if (out_pixel_count == 4096) begin 
                    // Ch?t th?i ?i?m pixel cu?i cùng ra kh?i m?ch
                    end_time = $time;
                    
                    // Tính toán các thông s? hi?u n?ng
                    elapsed_time_ns = end_time - start_time;
                    total_cycles    = elapsed_time_ns / 10.0;
                    patch_time_ms   = elapsed_time_ns / 1000000.0;
                    frame_time_ms   = patch_time_ms * 16.0; // 1 Khung hình chu?n g?m 16 patches
                    fps_val         = 1000.0 / frame_time_ms;

                    $fclose(file_out_id);

                    // IN B?NG BÁO CÁO TR?C TI?P LÊN TCL CONSOLE
                    $display("\n=======================================================");
                    $display("       BÁO CÁO ?O ??C HI?U N?NG PH?N C?NG TH?C T?       ");
                    $display("=======================================================");
                    $display("[*] T?ng s? ?i?m ?nh x? lý : %0d pixels", out_pixel_count);
                    $display("[*] Th?i ?i?m b?t ??u       : %0t ns", start_time);
                    $display("[*] Th?i ?i?m hoàn t?t      : %0t ns", end_time);
                    $display("-------------------------------------------------------");
                    $display("[*] T?ng chu k? (1 Patch)  : %0d CLOCK CYCLES", total_cycles);
                    $display("[*] Th?i gian ch?y (1 Patch): %0.4f ms (%0.2f ns)", patch_time_ms, elapsed_time_ns);
                    $display("[*] Th?i gian 1 Frame (x16) : %0.4f ms", frame_time_ms);
                    $display("[*] T?c ?? khung hình (FPS) : %0.2f FPS", fps_val);
                    $display("=======================================================\n");

                    $finish; 
                end
            end
        end
    end
endmodule
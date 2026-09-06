`timescale 1ns / 1ps

module tb_axis_smoke;
    reg clk;
    reg rst_n;
    reg [31:0] s_data;
    reg s_valid;
    wire s_ready;
    reg s_last;
    wire [31:0] m_data;
    wire m_valid;
    reg m_ready;
    wire m_last;
    wire protocol_error;
    integer cycle_count;
    integer output_count;

    srcnn_axis_wrapper #(
        .IMAGE_WIDTH(2),
        .IMAGE_HEIGHT(2),
        .WORDS_PER_PATCH(1),
        .FIFO_DEPTH(4),
        .FIFO_ADDRESS_BITS(2),
        .WEIGHT_FILE("smoke_weights.txt"),
        .BIAS_FILE("smoke_biases.txt")
    ) dut (
        .aclk(clk),
        .aresetn(rst_n),
        .s_axis_tdata(s_data),
        .s_axis_tvalid(s_valid),
        .s_axis_tready(s_ready),
        .s_axis_tlast(s_last),
        .m_axis_tdata(m_data),
        .m_axis_tvalid(m_valid),
        .m_axis_tready(m_ready),
        .m_axis_tlast(m_last),
        .protocol_error(protocol_error)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        s_data = 32'hFF804000;
        s_valid = 1'b0;
        s_last = 1'b0;
        m_ready = 1'b0;
        cycle_count = 0;
        output_count = 0;
        repeat (8) @(posedge clk);
        rst_n <= 1'b1;
        repeat (3) @(posedge clk);
        s_valid <= 1'b1;
        s_last <= 1'b1;
        while (!(s_valid && s_ready))
            @(posedge clk);
        @(posedge clk);
        s_valid <= 1'b0;
        s_last <= 1'b0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
            m_ready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            m_ready <= ((cycle_count % 7) != 0) && ((cycle_count % 11) != 0);
            if (m_valid && m_ready) begin
                output_count <= output_count + 1;
                if (m_data !== 32'h80808080) begin
                    $display("FAIL: output data %h", m_data);
                    $finish;
                end
                if (!m_last) begin
                    $display("FAIL: missing TLAST");
                    $finish;
                end
                if (protocol_error) begin
                    $display("FAIL: protocol_error asserted");
                    $finish;
                end
                $display("PASS: one word, four pixels, TLAST correct, cycles=%0d", cycle_count);
                $finish;
            end
            if (cycle_count > 5000) begin
                $display("FAIL: timeout, outputs=%0d", output_count);
                $finish;
            end
        end
    end

endmodule

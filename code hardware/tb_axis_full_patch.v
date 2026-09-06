`timescale 1ns / 1ps

// Full 128x128 AXI framing test with zero coefficients.
// It proves 4096 accepted input words produce exactly 4096 output words.
module tb_axis_full_patch;

    reg         aclk;
    reg         aresetn;
    reg [31:0]  s_axis_tdata;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tlast;
    wire [31:0] m_axis_tdata;
    wire        m_axis_tvalid;
    reg         m_axis_tready;
    wire        m_axis_tlast;
    wire        protocol_error;

    integer input_word_count;
    integer output_word_count;
    integer cycle_count;

    srcnn_axis_wrapper #(
        .IMAGE_WIDTH       (128),
        .IMAGE_HEIGHT      (128),
        .WORDS_PER_PATCH   (4096),
        .FIFO_DEPTH        (16),
        .FIFO_ADDRESS_BITS (4),
        .OUTPUT_ZERO_POINT (128),
        .WEIGHT_FILE       ("smoke_weights.txt"),
        .BIAS_FILE         ("smoke_biases.txt")
    ) dut (
        .aclk           (aclk),
        .aresetn        (aresetn),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tlast   (s_axis_tlast),
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (m_axis_tready),
        .m_axis_tlast   (m_axis_tlast),
        .protocol_error (protocol_error)
    );

    initial begin
        aclk = 1'b0;
        forever #5 aclk = ~aclk;
    end

    always @* begin
        s_axis_tdata = 32'h80808080;
        s_axis_tlast = (input_word_count == 4095);
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            input_word_count <= 0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            input_word_count <= input_word_count + 1;
            if (input_word_count == 4095)
                s_axis_tvalid <= 1'b0;
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            output_word_count <= 0;
            cycle_count <= 0;
            m_axis_tready <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            m_axis_tready <= ((cycle_count % 13) != 3) &&
                             ((cycle_count % 17) != 5);

            if (cycle_count > 1000000) begin
                $display("FAIL: full-patch timeout, input=%0d output=%0d",
                         input_word_count, output_word_count);
                $finish;
            end

            if (m_axis_tvalid && m_axis_tready) begin
                if (m_axis_tdata !== 32'h80808080) begin
                    $display("FAIL: bad data at output word %0d: %08h",
                             output_word_count, m_axis_tdata);
                    $finish;
                end
                if (m_axis_tlast !== (output_word_count == 4095)) begin
                    $display("FAIL: bad TLAST at output word %0d: %0b",
                             output_word_count, m_axis_tlast);
                    $finish;
                end
                output_word_count <= output_word_count + 1;
                if (output_word_count == 4095) begin
                    if (input_word_count != 4096 || protocol_error) begin
                        $display("FAIL: input=%0d protocol_error=%0b",
                                 input_word_count, protocol_error);
                        $finish;
                    end
                    $display("PASS: 4096 input words -> 4096 output words; TLAST correct; cycles=%0d",
                             cycle_count);
                    $finish;
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        s_axis_tvalid = 1'b0;
        input_word_count = 0;
        output_word_count = 0;
        cycle_count = 0;
        repeat (8) @(posedge aclk);
        aresetn = 1'b1;
        @(negedge aclk);
        s_axis_tvalid = 1'b1;
    end

endmodule

`timescale 1ns / 1ps

// Bit-exact 4x4 functional test.
// The only non-zero coefficient in each layer is the spatial center path:
// Layer 1 channel 0 = 0.5, Layer 2 channel 0 = 0.5,
// Layer 3 channel 0 = 0.5. For centered input x=0..15, the expected
// output is 128 + floor(x/8).
module tb_axis_functional;

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

    integer sent_words;
    integer received_words;
    integer cycle_count;
    integer mismatch_count;
    reg [31:0] expected_word;

    srcnn_axis_wrapper #(
        .IMAGE_WIDTH       (4),
        .IMAGE_HEIGHT      (4),
        .WORDS_PER_PATCH   (4),
        .FIFO_DEPTH        (4),
        .FIFO_ADDRESS_BITS (2),
        .OUTPUT_ZERO_POINT (128),
        .WEIGHT_FILE       ("functional_weights.txt"),
        .BIAS_FILE         ("functional_biases.txt")
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
        case (sent_words)
            0: s_axis_tdata = 32'h83828180;
            1: s_axis_tdata = 32'h87868584;
            2: s_axis_tdata = 32'h8B8A8988;
            default: s_axis_tdata = 32'h8F8E8D8C;
        endcase
        s_axis_tlast = (sent_words == 3);
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            sent_words <= 0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            sent_words <= sent_words + 1;
            if (sent_words == 3)
                s_axis_tvalid <= 1'b0;
        end
    end

    // Periodic output stalls exercise FIFO push/pop and AXI backpressure.
    always @(posedge aclk) begin
        if (!aresetn)
            m_axis_tready <= 1'b0;
        else
            m_axis_tready <= ((cycle_count % 5) != 1);
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            received_words <= 0;
            cycle_count <= 0;
            mismatch_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (cycle_count > 5000) begin
                $display("FAIL: timeout");
                $finish;
            end

            if (m_axis_tvalid && m_axis_tready) begin
                expected_word = (received_words < 2) ? 32'h80808080 : 32'h81818181;
                $display("TRACE: word %0d expected=%08h actual=%08h last=%0b",
                         received_words, expected_word, m_axis_tdata, m_axis_tlast);
                if (m_axis_tdata !== expected_word) begin
                    $display("FAIL: word %0d expected=%08h actual=%08h",
                             received_words, expected_word, m_axis_tdata);
                    mismatch_count <= mismatch_count + 1;
                end
                if (m_axis_tlast !== (received_words == 3)) begin
                    $display("FAIL: TLAST on word %0d is %0b", received_words, m_axis_tlast);
                    $finish;
                end
                received_words <= received_words + 1;
                if (received_words == 3) begin
                    if (protocol_error) begin
                        $display("FAIL: protocol_error asserted");
                        $finish;
                    end
                    if (mismatch_count == 0 && m_axis_tdata === expected_word)
                        $display("PASS: bit-exact 1->16->8->1, four AXI words, TLAST, stalls; cycles=%0d",
                                 cycle_count);
                    else
                        $display("FAIL: bit-exact comparison completed with mismatches");
                    $finish;
                end
            end
        end
    end

    initial begin
        aresetn = 1'b0;
        s_axis_tvalid = 1'b0;
        sent_words = 0;
        received_words = 0;
        cycle_count = 0;
        mismatch_count = 0;
        repeat (8) @(posedge aclk);
        aresetn = 1'b1;
        // Drive AXI inputs away from the sampling edge to avoid a testbench race.
        @(negedge aclk);
        s_axis_tvalid = 1'b1;
    end

endmodule

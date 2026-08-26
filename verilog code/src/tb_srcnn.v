`timescale 1ns / 1ps

module tb_srcnn;

reg clk;
reg rst_n;

reg  [31:0] s_tdata;
reg         s_tvalid;
reg         s_tlast;
wire        s_tready;

wire [31:0] m_tdata;
wire        m_tvalid;
wire        m_tlast;
reg         m_tready;

reg [7:0] image [0:4095];

integer i;
integer file_out;
integer pixel_count;


// ================================================================
// DUT
// ================================================================
srcnn_axis_wrapper uut (
    .clk            (clk),
    .rst_n          (rst_n),

    .s_axis_tdata   (s_tdata),
    .s_axis_tvalid  (s_tvalid),
    .s_axis_tready  (s_tready),
    .s_axis_tlast   (s_tlast),

    .m_axis_tdata   (m_tdata),
    .m_axis_tvalid  (m_tvalid),
    .m_axis_tready  (m_tready),
    .m_axis_tlast   (m_tlast)
);


// Clock 100 MHz
always #5 clk = ~clk;


// ================================================================
// KICH BAN GUI DU LIEU VA FLUSH PIPELINE
// ================================================================
initial begin
    clk         = 0;
    rst_n       = 0;

    s_tdata     = 0;
    s_tvalid    = 0;
    s_tlast     = 0;

    m_tready    = 1;
    pixel_count = 0;

    $readmemh(
        "C:/Users/MSI/SRCNN_Hardware/input_image_64x64.txt",
        image
    );

    file_out = $fopen(
        "C:/Users/MSI/SRCNN_Hardware/output_image_64x64.txt",
        "w"
    );

    if (file_out == 0) begin
        $display("ERROR: KHONG THE MO FILE OUTPUT");
        $finish;
    end

    // Reset
    #20;
    rst_n = 1;

    #20;


    // ============================================================
    // GUI 4096 PIXEL THAT
    // ============================================================
    for (i = 0; i < 4096; i = i + 1) begin

        // Master thay doi tin hieu tai negedge.
        // DUT lay mau tai posedge.
        @(negedge clk);

        s_tvalid = 1'b1;
        s_tdata  = {24'b0, image[i]};

        if (i == 4095)
            s_tlast = 1'b1;
        else
            s_tlast = 1'b0;

        // Cho mot posedge co handshake thanh cong.
        @(posedge clk);

        while (s_tready !== 1'b1) begin
            @(posedge clk);
        end
    end


    // ============================================================
    // FLUSH PIPELINE BANG DUMMY PIXEL
    //
    // Sau pixel that thu 4096:
    //   s_tdata  = 0
    //   s_tvalid = 1
    //   s_tlast  = 0
    //
    // Tiep tuc bom dummy cho den khi nhan du 4096 output.
    // ============================================================
    @(negedge clk);

    s_tdata  = 32'b0;
    s_tvalid = 1'b1;
    s_tlast  = 1'b0;

    while (pixel_count < 4096) begin
        @(negedge clk);
    end


    // ============================================================
    // DA NHAN DU 4096 PIXEL OUTPUT
    // ============================================================
    s_tvalid = 1'b0;
    s_tdata  = 32'b0;
    s_tlast  = 1'b0;

    $display(
        "TEST FULL ANH THANH CONG! TONG SO PIXEL = %0d",
        pixel_count
    );

    $fclose(file_out);
    $finish;
end


// ================================================================
// NHAN DU LIEU OUTPUT
//
// Output chi duoc ghi khi:
// m_tvalid && m_tready tai posedge clk
// ================================================================
always @(posedge clk) begin
    if (!rst_n) begin
        pixel_count = 0;
    end
    else begin
        if (m_tvalid && m_tready) begin
            if (pixel_count < 4096) begin
                $fwrite(file_out, "%02X\n", m_tdata[7:0]);
                $fflush(file_out);

                pixel_count = pixel_count + 1;
            end
        end
    end
end

endmodule
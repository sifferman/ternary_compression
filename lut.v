
module lut (
    input [7:0] a,
    output [9:0] b
);

reg [9:0] mem [0:255];
initial $readmemh(`MEMH_FILENAME, mem);
assign b = mem[a];

endmodule

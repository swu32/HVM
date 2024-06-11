
def parseDNA():
    # TO use: seq = parseDNA()
    directory = "/Users/swu/Documents/MouseHCM/HSTC/genome_data/genome_assemblies_genome_fasta/ncbi-genomes-2022-12-01" \
                "/GCF_000005845.2_ASM584v2_genomic.txt"
    with open(directory) as f:
        STR = f.read()

    def split(word):
        return [char for char in word]

    IR = split(STR)
    seq = []
    for it in IR:
        if it == 'A':
            seq.append(1)
        if it == 'T':
            seq.append(2)
        if it == 'C':
            seq.append(3)
        if it == 'G':
            seq.append(4)
    seq = np.array(seq).reshape([len(seq), 1, 1])
    return seq

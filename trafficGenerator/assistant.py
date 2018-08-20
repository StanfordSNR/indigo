class Function:
    def __init__(self, expression, domain, definition):
        self.func_Mbps = expression
        self.lower_s = float(domain[0])
        self.upper_s = float(domain[1])
        self.definition_s = float(definition)

    def get_pacing_bin(self):
        pacing_bin = []
        curr_s = self.lower_s
        while curr_s < self.upper_s:
            abs_rate_bps = self.func_Mbps(curr_s) * 1000 * 1000
            pacing_bps = int(abs_rate_bps / 8) * 8
            pacing_bin.append(pacing_bps)
            curr_s += self.definition_s
        return pacing_bin

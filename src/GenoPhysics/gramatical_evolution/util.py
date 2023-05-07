def get_fit_cases(file_problem):
    f_in = open(file_problem, 'r')
    data = f_in.readlines()
    f_in.close()
    fit_cases_str = [case[:-1].split() for case in data[1:]]
    fit_cases = [[float(elem) for elem in case] for case in fit_cases_str]

    return fit_cases


def get_header(file_problem):
    f_in = open(file_problem)
    header_line = f_in.readline()[:-1]
    f_in.close()
    header_line = header_line.split()
    header = [int(header_line[0])] + [[[header_line[i], int(header_line[i + 1])]
                                       for i in range(1, len(header_line), 2)]]

    return header

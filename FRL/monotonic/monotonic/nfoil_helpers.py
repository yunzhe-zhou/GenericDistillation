import os
import string
import itertools

def get_nfoil_input_string_helper(xs, ys, x_names):
    
    def x_name_to_f_name(x_name):
        return 'f%s' % x_name

    def x_name_to_t_name(x_name):
        return 't%s' % x_name

    def i_to_i_name(i):
        return 'e%d' % i
            
    import io
    output = io.StringIO()

    output.write(\
"""\
classes(pos,neg).
type(class(k)).
"""\
)

    for x_name in x_names:
        output.write(\
"""\
type(%s(k,%s)).
""" % (x_name_to_f_name(x_name),x_name_to_t_name(x_name)))

    for x_name in x_names:
        output.write(\
"""\
rmode(%s(+,1)).
"""\
% x_name_to_f_name(x_name))
        
    for (i, (x_n, y_n)) in enumerate(zip(xs, ys)):
        output.write(\
"""\
%s(%s).
"""\
% ('pos' if y_n==0 else 'neg', i_to_i_name(i)))
        for (x_name, x_ni) in zip(x_names, x_n):
            if x_ni == 1:
                output.write(\
"""\
%s(%s,%d).
"""\
% (x_name_to_f_name(x_name), i_to_i_name(i), 1))
    output.flush()
    return output.getvalue()


def nfoil_output_to_rule_feature_indicies(lines, x_names):
    """
    lines is the output of readlines()
    """
    
    x_names_to_idx = {x_name:i for (i,x_name) in enumerate(x_names)}
    
    def f_name_to_x_name(f_name):
        return f_name[1:]
    
    def clause_string_to_x_name(s):
        return f_name_to_x_name(string.split(s,sep='(')[0])

    idx = 0

    rule_feature_indicies = []
    
    for l in reversed(lines):
        if l == 'Model Learned :\n':
            break
        elif (l != '\n' and l[0:9] != ' Accuracy'):
            clause_strings = l.split()[0:-1]
            rule_x_names = [clause_string_to_x_name(s) for s in clause_strings]


            rule_feature_indicies.append([x_names_to_idx[x_name] for x_name in rule_x_names])

    return rule_feature_indicies
            
    
def mine_nfoil_rules(xs, ys, x_names, input_file, output_file, nfoil_path, max_clause=2, random_seed=42, depth=1, max_rules=25):
    """
    xs should be list of lists/arrays - the binary feature vectors
    x_names is a list of corresponding feature names for xs
    ys should be a binary list - the 0/1 label
    input_file and output_file are locations of temp files that calling nfoil will generate, since nfoil runs by reading in a file and printing stuff to screen, which i pipe to another file
    nfoil_path is the path of the nfoil executable
    """
    
    input_string = get_nfoil_input_string_helper(xs, ys, x_names)
    f = open(input_file, 'w')
    f.write(input_string)
    f.close()

    cmd = '%s %s -C %d -k %d -b %d -H %d > %s' % (nfoil_path, input_file, max_clause, random_seed, depth, max_rules, output_file)

    os.system(cmd)

    f = open(output_file, 'r')
    output_lines = f.readlines()
    f.close()

    return nfoil_output_to_rule_feature_indicies(output_lines, x_names)

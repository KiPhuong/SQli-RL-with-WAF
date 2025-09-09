import re

line = " ORDER BY 1,SLEEP(5),BENCHMARK(1000000,MD5('A')),4,5,6,7,8,9,10,11,12,13,14-- "
tokens = re.findall(r"--.*?$|#.*?$|'|\"|\s+|\d+|[A-Za-z_][A-Za-z0-9_]*|[(),]|[^'\sA-Za-z0-9_(),]", line.rstrip('\n'))
print(tokens)
import re
def checksyn(mystr) :
    if mystr == 'p':
        return print('No Error')
    elif re.search('two\([abp],[abp]\)', mystr) == None and re.search('one\([abp]\)', mystr) == None:
        return print('Syntax Error')
    else :
        mystr = re.sub('two\([abp],[abp]\)', 'p', mystr)
        mystr = re.sub('one\([abp]\)', 'p', mystr)
        return checksyn(mystr)
checksyn(input())
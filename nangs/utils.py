def checkIsListOfStr(l):
    "Make sure that l is a list containing only strings"
    if isinstance(l, tuple):
        for i in l:
            if not isinstance(i, str):
                raise Exception(str(i) + ' must be a string')


def checkUnique(l):
    "Make sure that l does not contain repeated elements"
    for i, item1 in enumerate(l):
        for j, item2 in enumerate(l):
            if i != j and item1 == item2:
                raise Exception('Repeated item ' + str(item1))


def checkNoRepeated(l1, l2):
    "Make sure there are no repeated elements in both lists"
    for i in l1:
        if i in l2:
            raise Exception('Repeated item ' + str(i))

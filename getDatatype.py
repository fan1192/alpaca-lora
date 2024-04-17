from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.symbol import SymbolType
from ghidra.util.exception import DuplicateNameException
from ghidra.program.model.data import Structure
from ghidra.program.model.data import Composite
import json

program = getCurrentProgram()
program.setImageBase(toAddr(0), 0)
dtmanager = currentProgram.getDataTypeManager()

all_data = dtmanager.getAllDataTypes()

all_info = {}
for data in all_data:
    info = []
    str_bytes = []
    str_name = []
    info.append(data.getLength())
    if isinstance(data, Structure) == True:
        comps = data.getComponents()
        for comp in comps:
            str_bytes.append(comp.getLength())
            #str_name.append(comp.getFieldName())
            str_name.append(comp.getDataType().getName())

    info.append(str_bytes)
    info.append(str_name)

    all_info[data.getName()] = info

filename = program.getName() + ".json"

with open(filename, 'w') as fp:
    json.dump(all_info, fp)

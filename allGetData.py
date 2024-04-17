#TODO write a description for this script
#@author
#@category _NEW_
#@keybinding
#@menupath
#@toolbar


#TODO Add User Code Here

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.symbol import SymbolType
from ghidra.util.exception import DuplicateNameException
from ghidra.program.model.data import DataTypeConflictHandler
from ghidra.program.model.data import TypedefDataType
from ghidra.program.model.symbol import SourceType

import json
import signal

program = getCurrentProgram()
program.setImageBase(toAddr(0), 0)
function = getFirstFunction()
ifc = DecompInterface()
ifc.openProgram(program)
funcmngr = currentProgram.getFunctionManager()
dtmanager = currentProgram.getDataTypeManager()

origtype = None

print("///////////////////////////////////Beginning/////////////////////////////////")

file_dict = []
while function is not None:
    print("///////////////////////////////////New Function/////////////////////////////////")
    print(function.name)
    func_dict = {}
    orig_name = []
    answer = {}
    funcname = function.name
    orig_name.append(funcname)
    startaddr = function.getEntryPoint()
    func_count = 1
    function.setName("FUNC" + str(func_count), SourceType.USER_DEFINED)
    answer["FUNC" + str(func_count)] = funcname
    functype = function.getReturnType()
    typedef = TypedefDataType("TYPE0", function.getReturnType())
    new_datatype = typedef.copy(dtmanager)
    function.setReturnType(new_datatype, SourceType.USER_DEFINED)
    answer["TPYE0"] = functype.getName()
    func_count += 1

    called_funcs = function.getCalledFunctions(ConsoleTaskMonitor())

    # Function
    for called_func in called_funcs:
        called_funcname = called_func.name
        orig_name.append(called_funcname)
        called_func.setName("FUNC" + str(func_count), SourceType.USER_DEFINED)
        answer["FUNC" + str(func_count)] = called_funcname
        func_count += 1


    # Varaible name and type
    locals = function.getAllVariables()
    origtypes = []
    orignames = []
    local_count = 1

    for idx in range(len(locals)):
        if locals[idx].getSymbol() != None:
            origtype = locals[idx].getDataType()
            typedef = TypedefDataType("TYPE" + str(local_count), locals[idx].getDataType())
            new_datatype = typedef.copy(dtmanager)
            locals[idx].setDataType(new_datatype, SourceType.USER_DEFINED)
            answer["TYPE" + str(local_count)] = origtype.getName()
            origtypes.append(origtype)

            origname = locals[idx].getName()
            locals[idx].setName("VAR" + str(local_count), SourceType.USER_DEFINED)
            answer["VAR" + str(local_count)] = origname
            orignames.append(origname)

            local_count += 1

    results = ifc.decompileFunction(function, 3, ConsoleTaskMonitor())
    # if results.getDecompiledFunction() == None:
    #     continue
    if results.getDecompiledFunction() != None:
        dec_res = results.getDecompiledFunction().getC()
        dec_res = dec_res.encode('ascii', 'ignore')
        func_dict['funcbody'] = dec_res

    assembly = ""
    currentInstr = getInstructionContaining(startaddr)
    while currentInstr != None and getFunctionContaining(currentInstr.getAddress()) == function:
        assembly+= currentInstr.toString() + "\n"
        currentInstr = currentInstr.getNext()

    func_dict['assembly'] = assembly

    function.setName(orig_name[0], SourceType.USER_DEFINED)
    function.setReturnType(functype, SourceType.USER_DEFINED)
    i=1
    for called_func in called_funcs:
        called_func.setName(orig_name[i], SourceType.USER_DEFINED)
        i+=1

    i=0
    for idx in range(len(locals)):
        if locals[idx].getSymbol() != None:
            locals[idx].setDataType(origtypes[i], SourceType.USER_DEFINED)
            locals[idx].setName(orignames[i], SourceType.USER_DEFINED)
            i+=1

    func_dict['answer'] = answer
    file_dict.append(func_dict)
    function = getFunctionAfter(function)

with open(program.name + "_O0_input.json", 'w') as fp:
    for func in file_dict:
        json.dump(func, fp)

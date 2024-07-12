from jodellSdk.jodellSdkDemo import ClawEpgTool


if __name__ == "__main__":
    clawTool = ClawEpgTool()
    comlist = clawTool.searchCom()
    print(comlist)
    flag = clawTool.serialOperation("", 115200, True) # 连接
    print(flag)
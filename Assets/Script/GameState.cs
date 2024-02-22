using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;

public class GameState
{
    public static TcpClient tcpClient;
    public static NetworkStream dataStream;

    public static bool accelerateKeyDown = false;
    public static bool leftKeyDown = false;
    public static bool rightKeyDown = false;

    public static bool accelerate = false;
    public static int steerState = 0;

    public static bool gameEnd = false;
    public static bool gamePause = true;
}

public class Consts
{
    public const int Accelerate = 1;
    public const int Brake = 0;

    public const int Steer_Straight = 1;
    public const int Steer_Left = 0;
    public const int Steer_Right = 2;

    public const int Cmd_Client_Action = 0;
    public const int Cmd_Client_Reset = 1;

    public const int Cmd_Server_Train = 10;
    public const int Cmd_Server_SaveNN = 11;
    public const int Cmd_Server_Terminate = 12;
    public const int Cmd_Server_Selective_Train = 13;
    public const int Cmd_Server_ResetTraining = 14;

    public const float minCarDist = 1;
}

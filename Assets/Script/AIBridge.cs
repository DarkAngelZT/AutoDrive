using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class AIBridge : MonoBehaviour
{
    static public AIBridge instance;

    byte[] buffer = new byte[128];
    void Awake()
    {
        instance = this;        
    }
    private void Start()
    {
        
    }
    void SendMsg(ref byte[] data)
    {
        var stream = GameState.dataStream;
        stream.Write(data, 0, data.Length);
    }

    public void ConnectTorch()
    {
        if (LogicManager.instance.controlMode == ControlMode.AI)
        {
            GameState.tcpClient = new System.Net.Sockets.TcpClient();
            GameState.tcpClient.Connect("localhost", 6666);
            GameState.dataStream = GameState.tcpClient.GetStream();
            if (GameState.dataStream != null)
            {
                Debug.Log("Backend connected");
            }
            else
            {
                Debug.LogWarning("Fail to connect to backend");
            }
        }
    }

    void ReceiveMsg(ref byte[] buffer)
    {
        var stream = GameState.dataStream;
        stream.Read(buffer, 0, buffer.Length);
    }

    static int PackageVector3(ref byte[]buffer,int start,ref Vector3 vec)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(vec.x), 0, buffer, start, 4);
        //Buffer.BlockCopy(BitConverter.GetBytes(vec.y), 0, buffer, start+4, 4);
        Buffer.BlockCopy(BitConverter.GetBytes(vec.z), 0, buffer, start+4, 4);
        return start + 8;
    }

    static int PackSensorData(ref byte[]buffer, int start, ref float[]data)
    {
        for(int i=0;i<data.Length;i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(data[i]), 0, buffer, start, 4);
            start += 4;
        }
        return start;
    }

    static int PackFloat(ref byte[] buffer, int offset, ref float val)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(val), 0, buffer, offset, 4);
        return offset + 4;
    }

    static int PackInt(ref byte[] buffer, int offset, ref int val)
    {
        Buffer.BlockCopy(BitConverter.GetBytes(val), 0, buffer, offset, 4);
        return offset + 4;
    }

    static int UnpackageVector3(ref byte[]data, int offset, out Vector3 vec)
    {
        vec = new Vector3();
        vec.x = BitConverter.ToSingle(data, offset);
        vec.y = 0;
        vec.z = BitConverter.ToSingle(data, offset + 4);
        return offset + 8;
    }

    static int UnpackInt(ref byte[] data, int offset,out int val)
    {
        val = BitConverter.ToInt32(data,offset);
        return offset + 4;
    }

    private void OnDestroy()
    {
        if(GameState.dataStream!=null)
        {
            if(GameState.tcpClient.Connected)
                EndTrainning();
            GameState.tcpClient.Close();
            GameState.dataStream.Close();

            GameState.tcpClient = null;
            GameState.dataStream = null;

            Debug.Log("Backend disconnected");
        }
        instance = null;
    }

    public void ComTest()
    {
        Vector3[] mockSensorData = new Vector3[8];
        Vector3 speed = new Vector3(0, 0, 10);
        Vector3 direction = new Vector3(0.5f, 0, 0.5f);

        for(int i=0;i<8; i++)
        {
            mockSensorData[i] = new Vector3(i*0.5f, 0, i*0.5f);
        }

        byte[] bytes = new byte[80];
        for (int i = 0; i < 8; i++)
        {
            PackageVector3(ref bytes,i*8,ref mockSensorData[i]);
        }
        PackageVector3(ref bytes, 64, ref speed);
        PackageVector3(ref bytes, 72, ref direction);

        SendMsg(ref bytes);

        byte[] b_recv = new byte[64];
        ReceiveMsg(ref b_recv);
        int offset = 0;
        int cmd;
        int steer, acc;
        offset= UnpackInt(ref b_recv, offset, out cmd);
        if(cmd==0)
        {
            offset = UnpackInt(ref b_recv, offset, out steer);
            offset = UnpackInt(ref b_recv, offset, out acc);

            Debug.Log(string.Format("{0},[{1},{2}]", cmd, steer, acc));
        }
        
    }

    public void SaveNeuralNetwork()
    {
        int cmd = Consts.Cmd_Server_SaveNN;
        PackInt(ref buffer, 0, ref cmd);
        SendMsg(ref buffer);
    }

    public void EndTrainning()
    {
        int cmd = Consts.Cmd_Server_Terminate;
        PackInt(ref buffer, 0, ref cmd);
        SendMsg(ref buffer);
    }

    public void SendTrainingData(float[] sensorData, ref float angle, ref float trackPos, ref Vector3 speed, float angularSpeed, float reward, bool success)
    {
        int offset = 0;
        int cmd = Consts.Cmd_Server_Train;
        if(LogicManager.instance.selectiveTrain)
        {
            cmd = Consts.Cmd_Server_Selective_Train;
        }
        float angleRad = angle * Mathf.Deg2Rad;
        float rotRad = angularSpeed * Mathf.Deg2Rad;
        offset = PackInt(ref buffer, offset, ref cmd);
        offset = PackSensorData(ref buffer, offset, ref sensorData);
        offset = PackFloat(ref buffer, offset, ref angleRad);
        offset = PackFloat(ref buffer, offset, ref trackPos);
        offset = PackageVector3(ref buffer, offset, ref speed);
        offset = PackFloat(ref buffer, offset, ref rotRad);
        offset = PackFloat(ref buffer, offset, ref reward);
        int s = success ? 1 : 0;
        offset = PackInt(ref buffer, offset, ref s);
        SendMsg(ref buffer);
    }

    public void HandleServerData()
    {
        ReceiveMsg(ref buffer);
        int offset = 0;
        int cmd;
        offset= UnpackInt(ref buffer, offset, out cmd);
        if(cmd == Consts.Cmd_Client_Reset)
        {
            int isTraining;
            offset = UnpackInt(ref buffer, offset, out isTraining);
            LogicManager.instance.ResetGame();//isTraining>0);
            LogicManager.instance.OnAIResetGame();
        }
        else if(cmd == Consts.Cmd_Client_Action)
        {
            int steer, acc;
            offset = UnpackInt(ref buffer, offset, out steer);
            offset = UnpackInt(ref buffer, offset, out acc);
            LogicManager.instance.OnAIInput(steer, acc);
        }
    }
}

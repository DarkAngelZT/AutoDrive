using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public enum ControlMode
{
    Manual,
    AI
}

public enum GameMode
{
    Training,
    Infinite
}
public class LogicManager : MonoBehaviour
{
    public static LogicManager instance;

    [SerializeField]
    private Text episodeText;
    [SerializeField]
    private Text speedText;
    [SerializeField]
    private Text contorlModeText;

    [SerializeField]
    public ControlMode controlMode = ControlMode.Manual;

    [SerializeField]
    public GameMode gameMode = GameMode.Training;

    [SerializeField]
    float envUpdateInterval = 0.05f;

    public bool selectiveTrain = false;

    public float maxSpeed;
    public float minSpeed;
    public float brakeSpeed;
    public float angularSpeed;
    public float acc;
    public float roadSize = 10f;
    public float minBrakeSpeed = 4;

    [SerializeField]
    float gameSpeed = 1;
    int[] frontSensorIndex = new int[] { 0, 1, 2, 10, 11 };
    float[] frontSensor = new float[5];

    private int episode = 0;
    private float lastUpdateTime = -1;

    [SerializeField]
    Transform startPoint;

    [SerializeField]
    Transform[] SegmentStartPoints;

    [SerializeField]
    int segmentTrainEpisode = 330;

    public int currentSegment = 0;

    public Car car;

    private void Awake()
    {
        instance = this;
    }

    void Start()
    {
        if(controlMode == ControlMode.Manual)
        {
            contorlModeText.text = "手动控制";
        }
        else if(controlMode == ControlMode.AI)
        {
            contorlModeText.text = "AI控制";
        }

        if(gameSpeed>1)
        {
            envUpdateInterval /= gameSpeed;
            maxSpeed *= gameSpeed;
            brakeSpeed *= gameSpeed;
            angularSpeed *= gameSpeed;
            acc *= gameSpeed;
        }
        ResetGame();
    }

    // Update is called once per frame
    void Update()
    {
        if(controlMode == ControlMode.Manual)
        {
            GameState.accelerate = GameState.accelerateKeyDown;
            if(GameState.leftKeyDown)
            {
                if(!GameState.rightKeyDown)
                {
                    GameState.steerState = Consts.Steer_Left;
                }
                else
                {
                    GameState.steerState = Consts.Steer_Straight;
                }
            }
            else if(GameState.rightKeyDown)
            {
                GameState.steerState = Consts.Steer_Right;
            }
            else
            {
                GameState.steerState = Consts.Steer_Straight;
            }
        }
    }

    private void FixedUpdate()
    {
        speedText.text = string.Format("{0:0. m/s}",car.GetSpeedNum());
        if(controlMode == ControlMode.AI)
        {
            if(lastUpdateTime<0)
            {
                lastUpdateTime = Time.time;
                if (gameMode == GameMode.Infinite)
                {
                    MapManager.instance.GenerateMap(); 
                }
                car.UpdateRoadInfo();
                AIBridge.instance.ConnectTorch();                
                AIBridge.instance.HandleServerData();                
            }
            else if (Time.time - lastUpdateTime > envUpdateInterval)
            {
                lastUpdateTime = Time.time;
                car.UpdateRoadInfo();
                Vector3 speed = car.GetSpeed();
                float[] sensor = car.GetSensorData();
                float roadDistance, roadAngle;
                car.GetRoadInfo(out roadDistance, out roadAngle);
                float reward = CalculateReward(sensor, roadDistance,ref speed, ref roadAngle);
                ep_reward += reward;
                Time.timeScale = 0;
                AIBridge.instance.SendTrainingData(sensor, ref roadAngle, ref roadDistance, ref speed, car.GetAngularSpeed(), reward, false);
                AIBridge.instance.HandleServerData();
                Time.timeScale = 1;
            }            
        }        
    }

    public void ResetGame(bool addNoise=false)
    {
        Transform startPos = SegmentStartPoints[currentSegment];
        if(car!=null)
        {
            Vector3 pos = startPos.position;
            Quaternion rot = startPos.rotation;
            car.ResetState();
            if (addNoise)
            {
                //加一点偏移，尽可能探索所有的进入位置
                pos.x += Random.Range( - roadSize * 0.33f, roadSize * 0.33f);
                //再加入一些随机旋转
                float angle = Random.Range(-30f, 30f);
                car.transform.localRotation = Quaternion.AngleAxis(angle, Vector3.up) * rot;
            }            
            car.transform.localPosition = pos;
        }
        GameState.gameEnd = false;
        GameState.gamePause = true;
    }

    public void OnAIResetGame()
    {
        car.UpdateRoadInfo();
        Vector3 speed = car.GetSpeed();
        float[] sensor = car.GetSensorData();
        float roadDistance, roadAngle;
        car.GetRoadInfo(out roadDistance, out roadAngle);
        AIBridge.instance.SendTrainingData(sensor, ref roadAngle, ref roadDistance, ref speed, car.GetAngularSpeed(), -1, false);
        episode += 1;
        ep_reward = 0;
        episodeText.text = episode.ToString();
        GameState.gamePause = false;
        AIBridge.instance.HandleServerData();
        selectiveTrain = false;
        //if (episode > 300)
        //{
        //    AIBridge.instance.EndTrainning();
        //    UnityEditor.EditorApplication.isPlaying = false;
        //}
    }
    int reach_count = 0;
    int lastEpisode = 0;
    float ep_reward = 0;
    public void OnReachDestination()
    {
        if (GameState.gameEnd) return;
        GameState.gameEnd = true;
        if (controlMode == ControlMode.AI)
        {
            car.UpdateRoadInfo();
            Vector3 speed = car.GetSpeed();
            float[] sensor = car.GetSensorData();
            float roadDist, roadAngle;
            car.GetRoadInfo(out roadDist, out roadAngle);
            float r = CalculateReward(sensor,roadDist, ref speed, ref roadAngle);
            r += 10;
            ep_reward += r;
            //睡觉前自行训练到            
            reach_count++;
            bool needCapture = false;
            if(needCapture && reach_count>9 && episode - lastEpisode>1)
            {
                lastEpisode = episode;
                AIBridge.instance.SaveNeuralNetwork();
                print(string.Format("Store episode {0}, reward: {1}, ep_reward: {2}",episode,r,ep_reward));
                if (reach_count > 29)
                {
                    AIBridge.instance.EndTrainning();
#if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
#endif
                    return;
                }
            }
            
            /*float bonuseBase = Mathf.Min(0.1f, Mathf.Abs(r * 0.1f));
            float bonus = Mathf.Abs(bonuseBase * speed.z / maxSpeed * (1 - roadDist));//bonus reward
            r += bonus;
            float r = Mathf.Abs(0.1f * speed.z / maxSpeed*(1-roadDist));*/
            
            AIBridge.instance.SendTrainingData(sensor, ref roadAngle, ref roadDist, ref speed, car.GetAngularSpeed(), r, true);
            AIBridge.instance.HandleServerData();
            
        }
    }

    public void OnHitWall()
    {
        if (GameState.gameEnd) return;
        GameState.gameEnd = true;
        if (controlMode == ControlMode.AI)
        {
            car.UpdateRoadInfo();
            Vector3 speed = car.GetSpeed();
            float[] sensor = car.GetSensorData();
            float roadDistance, roadAngle;
            car.GetRoadInfo(out roadDistance, out roadAngle);
            roadDistance = roadDistance >0? 1.01f : -1.01f;
            float r = 0;// CalculateReward(roadDistance, ref speed, ref roadAngle, true);
            ep_reward += r;
            AIBridge.instance.SendTrainingData(sensor, ref roadAngle, ref roadDistance, ref speed, car.GetAngularSpeed(), r, true);
            AIBridge.instance.HandleServerData();
        }
    }

    public void OnAIInput(int steer, int accelerate)
    {
        if (GameState.gameEnd)
            return;

        switch (accelerate)
        {
            case Consts.Accelerate:
                GameState.accelerate = true;
                break;
            case Consts.Brake:
            default:
                GameState.accelerate = false;
                break;
        }

        GameState.steerState = steer;

        //防止减速到0
        /*if(accelerate == Consts.Brake && car.GetSpeedNum()<minBrakeSpeed)
        {
            //如果速度降到最小速度以下还在踩刹车，就给与惩罚并且重新开始
            Vector3 speed = car.GetSpeed();
            float[] sensor = car.GetSensorData();
            float roadDistance, roadAngle;
            car.GetRoadInfo(out roadDistance, out roadAngle);
            roadDistance = roadDistance > 0 ? 1.01f : -1.01f;
            float r = -3;
            ep_reward += r;
            AIBridge.instance.SendTrainingData(sensor, ref roadAngle, ref roadDistance, ref speed, car.GetAngularSpeed(), r, true);
            AIBridge.instance.HandleServerData();
        }*/
    }

    float CalculateReward(float[]sensors,float centerDistance,ref Vector3 speed, ref float angle,bool hit=false)
    {
        float d = Mathf.Abs(centerDistance);
        //float minDist = Mathf.Min(sensors)/roadSize;
        float vz = speed.z;
        float vx = speed.x;
        float distanceFactor = Mathf.Abs(vz) > 0 ? 1 : 0;
        for(int i =0;i<frontSensorIndex.Length;i++)
        {
            frontSensor[i] = sensors[frontSensorIndex[i]];
        }
        float closestDist = Mathf.Min(frontSensor);
        //vz2 *= Mathf.Abs(vz2);
        //vx2 *= Mathf.Abs(vx2);

        float r =
            /*vz * Mathf.Cos(angle * Mathf.Deg2Rad)*/ distanceFactor*closestDist;
        //old formula
        //float base1 = d/0.6f;
        //float vz2 = (1-speed.z / maxSpeed),
        //    vx2 = speed.x / maxSpeed;
        //float vx = vx2* Mathf.Sin(angle * Mathf.Deg2Rad);
        //vz2 *= Mathf.Cos(angle * Mathf.Deg2Rad);
        //float escapePanalty = vx * centerDistance;
        //float speedFac = vz2 - vx2;
        //float angularFac = angle / 45f;
        //float adjustFac = angularFac * centerDistance;
        //angularFac *= angularFac;
        //float r = speedFac -(1- minDist/0.5f);
        return r;
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Car : MonoBehaviour
{
    Rigidbody body;
    float speed;
    Vector3 angularSpeed;
    Vector3 forward;
    const float epsil = 1e-3f;

    const int sensorCount = 12;
    float[] sensorData = new float[sensorCount];

    public const float sensorDistance = 50f;
    [SerializeField]
    Transform[] sensors;

    private Vector3 roadDirection = new Vector3();
    private float guideLineDistance = 0;

    int wallLayer;
    int guideLineLayer;
    int guideLineLayerMask;
    float guideLineDetectRadius = 6;

    bool hitWall = false;
    Animator animator;

    // Start is called before the first frame update
    void Start()
    {
        body = GetComponent<Rigidbody>();
        animator = GetComponent<Animator>();
        wallLayer = LayerMask.NameToLayer("Wall");
        guideLineLayer = LayerMask.NameToLayer("GuideLine");
        guideLineLayerMask = 1 << guideLineLayer;
        guideLineDetectRadius = (LogicManager.instance.roadSize + 2) * 0.5f;
    }

    private void Update()
    {
        //有可能同时碰到多个墙，导致重复发消息，所以就先改标志位，再下一帧统一处理
        if(hitWall)
        {
            LogicManager.instance.OnHitWall();
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if(!GameState.gamePause && !GameState.gameEnd)
        {
            speed = body.velocity.magnitude;
            angularSpeed = body.angularVelocity;
            float rotationFactor = speed / LogicManager.instance.maxSpeed;
            switch (GameState.steerState)
            {
                case Consts.Steer_Left:
                    if (speed > epsil)
                    {
                        angularSpeed.y = -Mathf.Deg2Rad * LogicManager.instance.angularSpeed * rotationFactor;
                        body.angularVelocity = angularSpeed;
                    }
                    else
                    {
                        angularSpeed.y = 0;
                        body.angularVelocity = angularSpeed;
                    }
                    animator.SetInteger("dir", -1);
                    break;
                case Consts.Steer_Right:
                    if (speed > epsil)
                    {
                        angularSpeed.y = Mathf.Deg2Rad * LogicManager.instance.angularSpeed * rotationFactor;
                        body.angularVelocity = angularSpeed;
                    }
                    else
                    {
                        angularSpeed.y = 0;
                        body.angularVelocity = angularSpeed;
                    }
                    animator.SetInteger("dir", 1);
                    break;
                case Consts.Steer_Straight:
                default:
                    if (Mathf.Abs(angularSpeed.y) > epsil)
                    {
                        angularSpeed.y = 0;
                        body.angularVelocity = angularSpeed;
                    }
                    animator.SetInteger("dir", 0);
                    break;
            }
            
            forward = transform.forward;
            if(GameState.accelerate)
            {
                if(speed< LogicManager.instance.maxSpeed)
                {
                    speed += LogicManager.instance.acc*Time.fixedDeltaTime;                                        
                }
                speed = Mathf.Clamp(speed, 0, LogicManager.instance.maxSpeed);
                body.velocity = forward * speed;
            }
            else
            {
                if(speed> LogicManager.instance.minSpeed)
                {
                    speed -= LogicManager.instance.brakeSpeed*Time.fixedDeltaTime;
                    speed = Mathf.Clamp(speed, 0, LogicManager.instance.maxSpeed);
                    speed = Mathf.Max(LogicManager.instance.minSpeed, speed);

                    body.velocity = forward * speed;
                }else
                {                  
                    speed += LogicManager.instance.acc * Time.fixedDeltaTime;
                    speed = Mathf.Clamp(speed, 0, LogicManager.instance.maxSpeed);
                    body.velocity = forward * speed;
                }
            }
        }
    }

    public void ResetState()
    {
        if (body == null) return;
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        transform.rotation = Quaternion.identity;
        speed = 0;
        hitWall = false;
    }

    public void UpdateRoadInfo()
    {
        Collider[] lines = Physics.OverlapSphere(transform.position, guideLineDetectRadius, guideLineLayerMask);
        roadDirection = Vector3.forward;
        guideLineDistance = 0;
        Vector3 pos = transform.position;
        float distance,closestDist=float.MaxValue;
        Vector3 linePos;
        if (lines.Length > 0)
        {
            roadDirection = Vector3.zero;
            foreach (var line in lines)
            {
                roadDirection += line.transform.forward;
                linePos = line.ClosestPoint(pos);
                distance = Vector3.Distance(linePos, pos);
                if (distance < closestDist)
                {
                    closestDist = distance;
                    guideLineDistance = distance;
                    roadDirection = line.transform.forward;
                }
            }
            roadDirection /= (float)lines.Length;
        }
    }

    public float[] GetSensorData()
    {
        bool hit;
        LayerMask wallMask = LayerMask.GetMask("Wall");
        RaycastHit hitInfo;
        for(int i=0;i< sensorCount; i++)
        {
            Transform s = sensors[i];
            hit = Physics.Raycast(s.position, s.forward, out hitInfo, sensorDistance, wallMask);
            if(hit)
            {
                sensorData[i] = hitInfo.distance;
            }
            else
            {
                sensorData[i] = sensorDistance;
            }
        }
        return sensorData;
    }

    public Vector3 GetSpeed()
    {
        //计算相对速度
        Vector3 v = body.velocity;
        Vector3 dir = v.normalized;
        float speed = v.magnitude;
        Quaternion rotation = Quaternion.FromToRotation(Vector3.forward, roadDirection);
        Vector3 v_relative = rotation*dir;
        v_relative.Normalize();
        return v_relative * speed;
    }

    public float GetSpeedNum()
    {
        return body.velocity.magnitude;
    }

    public void GetRoadInfo(out float distance, out float angle)
    {
        distance = guideLineDistance/(LogicManager.instance.roadSize*0.5f);
        angle = Vector3.SignedAngle(roadDirection,transform.forward,Vector3.up);
        if(angle<0)
        {
            distance *= -1;
        }
    }

    public float GetAngularSpeed()
    {
        return body.angularVelocity.y;
    }

    private void OnTriggerEnter(Collider other)
    {
        if(other.tag == "Finish")
        {
            LogicManager.instance.OnReachDestination();
        }
        if (other.tag == "Checkpoint")
        {
            MapManager.instance.OnCarReachCheckpoint();
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == wallLayer)
        {
            hitWall = true;            
        }        
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RacingCamera : MonoBehaviour
{
    [SerializeField]
    Transform target;
    [SerializeField]
    float lerp = 0.5f;
    [SerializeField]
    float MoveLerp = 0.5f;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float destAngle = target.transform.eulerAngles.y;
        float curAngle = transform.rotation.eulerAngles.y;

        float delta = destAngle - curAngle;
        if(delta>180)
        {
            delta -= 360;
        }
        else if(delta< -180)
        {
            delta += 360;
        }
        float angle = delta * lerp * Time.fixedDeltaTime;

        var rot = transform.rotation.eulerAngles;
        rot.y = angle+curAngle;
        transform.rotation = Quaternion.Euler(rot);
        transform.position = Vector3.Slerp(transform.position, target.position,MoveLerp*Time.fixedDeltaTime);
    }
}

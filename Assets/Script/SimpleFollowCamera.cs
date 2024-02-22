using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleFollowCamera : MonoBehaviour
{
    Vector3 offset;
    [SerializeField]
    Transform target;
    // Start is called before the first frame update
    void Start()
    {
        offset = transform.position - target.position;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Vector3 pos = target.position + offset;
        transform.position = pos;
    }
}

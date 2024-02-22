using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SelectiveTrainingTrigger : MonoBehaviour
{
    [SerializeField]
    bool activate = true;

    private void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.tag == "Car")
        {
            LogicManager.instance.selectiveTrain = activate;
        }
    }
}

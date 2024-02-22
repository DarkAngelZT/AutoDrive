using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.P))
        {
            //send save cmd
            AIBridge.instance.SaveNeuralNetwork();
        }

        if(Input.GetKeyUp(KeyCode.Return))
        {
            GameState.gamePause = false;
        }
        if (Input.GetKeyUp(KeyCode.R))
        {
            LogicManager.instance.ResetGame();
            if (LogicManager.instance.controlMode == ControlMode.AI)
            {
                LogicManager.instance.OnAIResetGame();
            }
        }
        if (Input.GetKeyUp(KeyCode.Escape))
        {
            GameState.gameEnd = true;
            if (LogicManager.instance.controlMode == ControlMode.AI)
            { 
                AIBridge.instance.EndTrainning();
            }
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
            //Application.Quit();
        }
        // accelerate
        if (Input.GetKey(KeyCode.W))
        {
            GameState.accelerateKeyDown = true;
        }
        else if (Input.GetKeyUp(KeyCode.W))
        {
            GameState.accelerateKeyDown = false;
        }
        //left right
        if (Input.GetKeyDown(KeyCode.A))
        {
            GameState.leftKeyDown = true;
        }
        else if (Input.GetKeyUp(KeyCode.A))
        {
            GameState.leftKeyDown = false;
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            GameState.rightKeyDown = true;
        }
        else if (Input.GetKeyUp(KeyCode.D))
        {
            GameState.rightKeyDown = false;
        }
    }
}

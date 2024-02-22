using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MapManager : MonoBehaviour
{
    public static MapManager instance;

    [SerializeField]
    Transform mapRoot;

    [SerializeField]
    GameObject[] tiles;
    [SerializeField]
    GameObject checkpoint;

    Dictionary<int, List<Tile>> objectPool = new Dictionary<int, List<Tile>>();

    List<GameObject> segmentTriggerPool = new List<GameObject>();

    Tile currentEndTile;

    [SerializeField]
    int tileCountPerSeg = 5;
    [SerializeField]
    int turnCoolDown = 2;
    [SerializeField]
    int sameDirCoolDown = 2;

    List<Tile> currentTiles = new List<Tile>();
    List<GameObject> currentCheckpoints = new List<GameObject>();
    Tile starterTile;

    int checkPointsCount = 0;

    int dirTrend = 0;
    int turnCd = 0;
    int sameSideCd = 0;
    List<int> straightIndices = new List<int>();
    List<int> leftTurnIndices = new List<int>();
    List<int> rightTurnIndices = new List<int>();

    private void Awake()
    {
        instance = this;
    }
    // Start is called before the first frame update
    void Start()
    {
        for(int i = 0;i<tiles.Length;i++)
        {
            objectPool.Add(i, new List<Tile>());
        }
        for(int k=0;k<tiles.Length;k++)
        {
            switch(tiles[k].GetComponent<Tile>().dirType)
            {
                case TileDirType.Left:
                    leftTurnIndices.Add(k);
                    break;
                case TileDirType.Right:
                    rightTurnIndices.Add(k);
                    break;
                case TileDirType.U_Shape:
                    if(tiles[k].GetComponent<Tile>().ushapeSide == UShapeSide.UShapeLeft)
                    {
                        leftTurnIndices.Add(k);
                    }
                    else
                    {
                        rightTurnIndices.Add(k);
                    }
                    break;
                case TileDirType.Straight:
                default:
                    straightIndices.Add(k);
                    break;
            }
        }
    }

    public void OnCarReachCheckpoint()
    {
        checkPointsCount++;
        CreateSegment();
        ClearOldSegment();
    }

    GameObject CreateTile(int type)
    {
        GameObject prototype = tiles[type].gameObject;
        GameObject go = Instantiate(prototype, mapRoot);
        go.GetComponent<Tile>().tileType = type;
        return go;
    }

    void RemoveTile(Tile t)
    {
        t.gameObject.SetActive(false);
        objectPool[t.tileType].Add(t);
    }

    Tile GetTile(int type)
    {
        Tile tile;
        if(objectPool[type].Count==0)
        {
            GameObject go = CreateTile(type);
            tile = go.GetComponent<Tile>();
        }
        else
        {
            int idx = objectPool[type].Count - 1;
            tile = objectPool[type][idx];
            objectPool[type].RemoveAt(idx);
            tile.gameObject.SetActive(true);
        }
        return tile;
        
    }

    GameObject GetCheckpoint()
    {
        GameObject go;
        if(segmentTriggerPool.Count==0)
        {
            go = Instantiate(checkpoint,mapRoot);
            return go;
        }
        else
        {
            int idx = segmentTriggerPool.Count - 1;
            go = segmentTriggerPool[idx];
            go.SetActive(true);
            segmentTriggerPool.RemoveAt(idx);
        }
        return go;
    }
    int DetermineTileType()
    {
        float t = Random.value;
        if(t<0.3f && turnCd==0)
        {
            //turn
            if(dirTrend==0)
            {
                int rt = Random.Range(0, leftTurnIndices.Count + rightTurnIndices.Count);
                if(rt<leftTurnIndices.Count)
                {
                    dirTrend = -1;
                    turnCd = turnCoolDown;
                    sameSideCd = sameDirCoolDown;
                    return leftTurnIndices[rt];
                }
                else
                {
                    dirTrend = 1;
                    turnCd = turnCoolDown;
                    sameSideCd = sameDirCoolDown;
                    return rightTurnIndices[rt - leftTurnIndices.Count];
                }
            }
            else
            {
                float side = Random.value;
                if(dirTrend==-1)
                {
                    if(sameSideCd==0 && side<0.5f)
                    {
                        int r = Random.Range(0, leftTurnIndices.Count);
                        dirTrend = -1;
                        turnCd = turnCoolDown;
                        sameSideCd = sameDirCoolDown;
                        return leftTurnIndices[r];
                    }
                    else
                    {
                        int r = Random.Range(0, rightTurnIndices.Count);
                        dirTrend = 1;
                        turnCd = turnCoolDown;
                        sameSideCd = sameDirCoolDown;
                        return rightTurnIndices[r];
                    }
                }
                else
                {
                    if (sameSideCd == 0 && side > 0.5f)
                    {
                        int r = Random.Range(0, rightTurnIndices.Count);
                        dirTrend = 1;
                        turnCd = turnCoolDown;
                        sameSideCd = sameDirCoolDown;
                        return rightTurnIndices[r];
                    }
                    else
                    {
                        int r = Random.Range(0, leftTurnIndices.Count);
                        dirTrend = -1;
                        turnCd = turnCoolDown;
                        sameSideCd = sameDirCoolDown;
                        return leftTurnIndices[r];
                    }
                }
            }
        }
        else
        {
            //straight
            int st = Random.Range(0, straightIndices.Count);
            turnCd = Mathf.Max(0, turnCd - 1);
            sameSideCd = Mathf.Max(0, sameSideCd - 1);
            return straightIndices[st];
        }
    }
    void CreateSegment()
    {
        for (int i = 0; i < tileCountPerSeg; i++)
        {
            Tile t = GetTile(DetermineTileType());
            currentTiles.Add(t);
            Transform jointPoint = currentEndTile.GetOutPoint();
            Transform jointPoint2 = t.GetInPoint();
            
            t.transform.rotation = jointPoint.rotation;

            Vector3 deltaPos = jointPoint.position - jointPoint2.position;
            Vector3 newPos = t.transform.position + deltaPos;
            t.transform.position = newPos;
            
            currentEndTile = t;
        }
        GameObject checkpoint = GetCheckpoint();
        Transform cTrans = checkpoint.transform;
        cTrans.position = currentEndTile.GetOutPoint().position;
        cTrans.rotation = currentEndTile.GetOutPoint().rotation;
        currentCheckpoints.Add(checkpoint);
    }

    void ClearOldSegment()
    {        
        if(checkPointsCount>1)
        {
            if (starterTile != null)
            {
                RemoveTile(starterTile);
                starterTile = null;
            }
            for (int i=0;i<tileCountPerSeg;i++)
            {
                RemoveTile(currentTiles[i]);
            }
            var go = currentCheckpoints[0];
            go.SetActive(false);
            segmentTriggerPool.Add(go);

            currentTiles.RemoveRange(0, tileCountPerSeg);
            currentCheckpoints.RemoveAt(0);
        }
    }

    public void GenerateMap()
    {
        if(currentEndTile==null)
        {
            starterTile = currentEndTile = GetTile(11);
            CreateSegment();
        }
        CreateSegment();
        ClearOldSegment();
    }
}

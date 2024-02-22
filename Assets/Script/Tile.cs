using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum TileDirType
{
    Straight,
    Left,
    Right,
    U_Shape
}

public enum UShapeSide
{
    UShapeLeft,
    UShapeRight
}
public class Tile : MonoBehaviour
{
    public int tileType;
    [SerializeField]
    public Transform in_point;
    [SerializeField]
    public Transform out_point;
    [SerializeField]
    public TileDirType dirType = TileDirType.Straight;
    [SerializeField]
    public float turningAngle = 0;
    [SerializeField]
    public UShapeSide ushapeSide = UShapeSide.UShapeLeft;
    [SerializeField]
    public Transform GuideLineRoot;

    private void Awake()
    {
        if(dirType == TileDirType.U_Shape)
        {
            if(ushapeSide== UShapeSide.UShapeLeft)
            {
                Collider[] lines = GuideLineRoot.GetComponentsInChildren<Collider>();
                var rotation = new Vector3(0, 180, 0);
                foreach (var l in lines)
                {
                    l.transform.Rotate(rotation);
                }
                in_point.Rotate(rotation);
                out_point.Rotate(rotation);
            }
        }
    }

    public Transform GetInPoint()
    {
        if(dirType == TileDirType.U_Shape)
        {
            if(ushapeSide== UShapeSide.UShapeLeft)
            {
                return out_point;
            }
            else
            {
                return in_point;
            }
        }
        else
        {
            return in_point;
        }
    }

    public Transform GetOutPoint()
    {
        if (dirType == TileDirType.U_Shape)
        {
            if (ushapeSide == UShapeSide.UShapeLeft)
            {
                return in_point;
            }
            else
            {
                return out_point;
            }
        }
        else
        {
            return out_point;
        }
    }
}

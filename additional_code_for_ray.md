## C# (Unity)
```csharp
private RayPerceptionSensorComponent3D raySensor;

public override void Initialize()
{
    raySensor = GetComponentInChildren<RayPerceptionSensorComponent3D>();
}

public override void OnActionReceived(ActionBuffers actions)
{
    var rays = raySensor.RaySensor.RayPerceptionOutput.RayOutputs;
    foreach (var ray in rays)
    {
        if (ray.HasHit)
        {
            AddReward(-0.1f * (1f - ray.HitFraction));
        }
    }   
}
```

## Python
```python
state_dim1 = spec.observation_specs[0].shape[0]
state_dim2 = spec.observation_specs[1].shape[0]
state_dim = state_dim1 + state_dim2

states_tensor = torch.from_numpy(np.concatenate(decision_steps.obs, axis=1)).to(torch.float32)
t_states_tensor = torch.from_numpy(np.concatenate(t_decision_steps.obs, axis=1)).to(torch.float32)

next_state = np.concatenate(next_terminal_steps[agent_id].obs, axis=0)
next_state = np.concatenate(next_decision_steps[agent_id].obs, axis=0)
```

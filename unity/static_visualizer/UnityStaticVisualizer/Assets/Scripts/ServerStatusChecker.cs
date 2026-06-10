using UnityEngine;
using UnityEngine.Networking;
using System.Threading.Tasks;

public class ServerStatusChecker : MonoBehaviour
{
    public string serverUrl = null;

    public async Task SetServerUrl()
    {
        string serverUrlLocal = await GetNGrokEndpoint.GetEndpoint(); 
        this.serverUrl = serverUrlLocal;
        await CheckServerStatus();
    }

    async Task CheckServerStatus()
    {
        if (string.IsNullOrEmpty(serverUrl))
        {
            Debug.LogError("Server URL is empty.");
            return;
        }

        bool reachable = await IsServerReachable(serverUrl);

        if (reachable)
        {
            Debug.Log("✅ Server reachable: " + serverUrl);
        }
        else
        {
            Debug.LogWarning("Server unreachable: " + serverUrl);
        }
    }

    async Task<bool> IsServerReachable(string url)
    {
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            request.timeout = 5;
            var operation = request.SendWebRequest();
            while (!operation.isDone)
                await Task.Yield();

#if UNITY_2020_1_OR_NEWER
            return request.result != UnityWebRequest.Result.ConnectionError &&
                   request.result != UnityWebRequest.Result.ProtocolError;
#else
            return !request.isNetworkError && !request.isHttpError;
#endif
        }
    }

    async void Start()
    {
         await SetServerUrl();
    }
}
    
using UnityEngine;
using System.Threading.Tasks;
using UnityEngine.Networking;
using System;

[Serializable]
public class EndpointData
{
    public string endpoint;
    public string last_checked;
}


public class GetNGrokEndpoint
{
    private const string GistUrl =
        "https://gist.githubusercontent.com/KahnSvaer/3c709360165ff302c242a38407cb03ad/raw/Endpoint.json";

    public static async Task<string> GetEndpoint()
    {
        string url =
        $"{GistUrl}?t={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"; // To avoid caching issues
        using UnityWebRequest request = UnityWebRequest.Get(url);

        var operation = request.SendWebRequest();

        while (!operation.isDone)
            await Task.Yield();

        if (request.result != UnityWebRequest.Result.Success)
            throw new Exception(request.error);

        EndpointData data =
            JsonUtility.FromJson<EndpointData>(
                request.downloadHandler.text);

        Debug.Log("Endpoint: " + data.endpoint + "Last Checked: " + data.last_checked);

        return data.endpoint;
    }
}

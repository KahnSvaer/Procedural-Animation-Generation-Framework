using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using System.Collections.Generic;
using System.Collections;

public class UIInteraction : MonoBehaviour
{
    [SerializeField] private GameObject welcomePanel;
    [SerializeField] private GameObject tutorialPanel;
    [SerializeField] private Button lightButton;
    [SerializeField] private Image lightOnImage;
    [SerializeField] private Image lightOffImage;
    [SerializeField] private Button hintButton;
    [SerializeField] private Button serverResetButton;

    [Header("Spawn Input UI")]
    [SerializeField] private TMP_InputField spawnInputField;
    [SerializeField] private Button spawnSubmitButton;

    [SerializeField] private Light helperLight;

    private bool lightOn = true;
    private List<Button> allButtons = new List<Button>();

    private float panelCooldown = 0.1f; // Added to avoid double toggles
    private float lastTutorialCloseTime = -1f;

    private Camera mainCamera;

    private ServerStatusChecker serverChecker;

    void Start()
    {
        if (welcomePanel) welcomePanel.SetActive(true);
        if (tutorialPanel) tutorialPanel.SetActive(false);
        if (lightButton) lightButton.gameObject.SetActive(false);
        if (hintButton) hintButton.gameObject.SetActive(false);
        if (serverResetButton) serverResetButton.gameObject.SetActive(false);
        if (spawnSubmitButton) spawnSubmitButton.gameObject.SetActive(false);
        if (spawnInputField) spawnInputField.gameObject.SetActive(false);

        if (lightButton) lightButton.onClick.AddListener(ToggleLight);
        if (hintButton) hintButton.onClick.AddListener(ShowTutorial);
        if (serverResetButton) serverResetButton.onClick.AddListener(ServerReset);

        if (spawnSubmitButton) spawnSubmitButton.onClick.AddListener(OnSpawnSubmit);

        lightOn = true;
        if (helperLight) helperLight.enabled = true;
        UpdateLightUI();

        if (lightButton) allButtons.Add(lightButton);
        if (hintButton) allButtons.Add(hintButton);
        if (serverResetButton) allButtons.Add(serverResetButton);
        if (spawnSubmitButton) allButtons.Add(spawnSubmitButton);

        mainCamera = Camera.main;

        serverChecker = GetComponent<ServerStatusChecker>();

    }

    void Update()
    {
        if (Keyboard.current != null)
        {
            if (Keyboard.current.enterKey.wasPressedThisFrame || Keyboard.current.spaceKey.wasPressedThisFrame)
            {
                if (welcomePanel.activeSelf) CloseWelcome();
                else if (tutorialPanel.activeSelf) HideTutorial();
            }
        }

        if (Mouse.current != null && Mouse.current.leftButton.wasPressedThisFrame)
        {
            if (welcomePanel.activeSelf && !IsPointerOverUI(welcomePanel)) CloseWelcome();
            else if (tutorialPanel.activeSelf && !IsPointerOverUI(tutorialPanel)) HideTutorial();
        }
    }

    private void ToggleLight()
    {
        lightOn = !lightOn;
        if (helperLight) helperLight.enabled = lightOn;
        UpdateLightUI();
    }

    private void UpdateLightUI()
    {
        if (lightOnImage) lightOnImage.enabled = lightOn;
        if (lightOffImage) lightOffImage.enabled = !lightOn;
    }

    private void ShowTutorial()
    {
        if (tutorialPanel.activeSelf) return;
        if (Time.time - lastTutorialCloseTime < panelCooldown) return;

        tutorialPanel.SetActive(true);
        BlockAllButtons(true);
    }

    private void HideTutorial()
    {
        if (!tutorialPanel.activeSelf) return;
        tutorialPanel.SetActive(false);
        StartCoroutine(UnlockButtonsWithDelay());
        lastTutorialCloseTime = Time.time;
    }

    private void CloseWelcome()
    {
        if (!welcomePanel.activeSelf) return;
        welcomePanel.SetActive(false);

        if (lightButton) lightButton.gameObject.SetActive(true);
        if (hintButton) hintButton.gameObject.SetActive(true);
        if (spawnSubmitButton) spawnSubmitButton.gameObject.SetActive(true);
        if (spawnInputField) spawnInputField.gameObject.SetActive(true);
        if (serverResetButton) serverResetButton.gameObject.SetActive(true);
        
        StartCoroutine(UnlockButtonsWithDelay());
    }
    private void OnSpawnSubmit()
    {
        if (!spawnInputField) return;
        string value = spawnInputField.text.Trim();
        Debug.Log("Spawn submitted: " + value);

        Vector3 pivot = mainCamera.GetComponent<InteractionManager>().pivotPoint;
        GetComponent<RuntimeGLTFUnzipAndApply>().Spawner(value, pivot, Quaternion.LookRotation(mainCamera.transform.position - pivot));
        spawnInputField.text = string.Empty;
    }

    private bool IsPointerOverUI(GameObject panel)
    {
        if (!panel) return false;
        PointerEventData eventData = new PointerEventData(EventSystem.current);
        eventData.position = Mouse.current.position.ReadValue();
        var results = new List<RaycastResult>();
        EventSystem.current.RaycastAll(eventData, results);
        foreach (var r in results)
            if (r.gameObject == panel || r.gameObject.transform.IsChildOf(panel.transform))
                return true;
        return false;
    }

    public void BlockAllButtons(bool block)
    {
        foreach (var btn in allButtons)
        {
            if (!btn) continue;
            btn.interactable = !block;
        }
    }

    private async void ServerReset()
    {
        if (serverChecker == null)
        {
            Debug.LogError("ServerStatusChecker component not found on UIInteraction object.");
            return;
        }
        await serverChecker.SetServerUrl();
    }

    private IEnumerator UnlockButtonsWithDelay()
    {
        yield return new WaitForSeconds(panelCooldown);
        BlockAllButtons(false);
    }
}

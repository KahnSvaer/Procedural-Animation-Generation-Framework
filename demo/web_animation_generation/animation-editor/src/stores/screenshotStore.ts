import { create } from "zustand";

interface ScreenshotStore {
    screenshotVersion: number;

    requestScreenshot: () => void;
}

export const useScreenshotStore = create<ScreenshotStore>()(
    (set) => ({
        screenshotVersion: 0,

        requestScreenshot: () =>
            set((state) => ({
                screenshotVersion:
                    state.screenshotVersion + 1,
            })),
    })
);
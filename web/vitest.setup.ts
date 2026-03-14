import '@testing-library/jest-dom/vitest';
import { expect, vi } from 'vitest';
import * as matchers from '@testing-library/jest-dom/matchers';

expect.extend(matchers);

// Create mock functions for play and pause
const mockPlay = vi.fn(() => Promise.resolve());
const mockPause = vi.fn();

// Spy on the prototype methods and replace with mocks
Object.defineProperty(window.HTMLMediaElement.prototype, 'play', {
    configurable: true,
    value: mockPlay,
});

Object.defineProperty(window.HTMLMediaElement.prototype, 'pause', {
    configurable: true,
    value: mockPause,
});

// Mock the Audio constructor
window.Audio = vi.fn().mockImplementation(() => ({
    play: mockPlay,
    pause: mockPause,
    // You might want to mock other properties like 'src', 'duration', 'paused'
    src: '',
    duration: NaN,
    paused: true,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
}));

// Export the mocks if you need to access them in individual tests for assertions
export { mockPlay, mockPause };

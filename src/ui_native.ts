// Native Windows Toast and Acrylic Context Menu System
// Designed to replace webby alerts and default browser context menus

export type ToastType = 'info' | 'success' | 'warning' | 'error';

const ICONS: Record<ToastType, string> = {
  success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`,
  error: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`,
  warning: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
  info: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`
};

const DEFAULT_TITLES: Record<ToastType, string> = {
  success: 'Success',
  error: 'Notice / Error',
  warning: 'Warning',
  info: 'Information'
};

function escapeHtml(str: string): string {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Shows a native-feeling dark acrylic toast notification in the bottom right corner.
 */
export function showToast(
  message: string,
  type: ToastType = 'info',
  customTitle?: string,
  duration = 3800
) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `native-toast toast-${type}`;
  const title = customTitle || DEFAULT_TITLES[type];

  toast.innerHTML = `
    <div class="toast-icon">${ICONS[type]}</div>
    <div class="toast-content">
      <div class="toast-title">${escapeHtml(title)}</div>
      <div class="toast-message">${escapeHtml(message)}</div>
    </div>
    <button class="toast-close" title="Dismiss">&times;</button>
  `;

  container.appendChild(toast);

  // Trigger smooth slide-in
  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  const dismiss = () => {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(() => {
      if (toast.parentNode === container) {
        container.removeChild(toast);
      }
    }, 250);
  };

  const closeBtn = toast.querySelector('.toast-close');
  closeBtn?.addEventListener('click', dismiss);

  if (duration > 0) {
    setTimeout(dismiss, duration);
  }
}

/**
 * Shows an elegant native dark confirmation modal dialog.
 */
export function showConfirm(
  title: string,
  message: string,
  confirmLabel = 'Confirm',
  isDestructive = true
): Promise<boolean> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-backdrop open';
    overlay.style.zIndex = '100002';

    const dialog = document.createElement('div');
    dialog.className = 'modal-dialog';
    dialog.style.width = '420px';

    dialog.innerHTML = `
      <div class="modal-header">
        <div class="modal-title">${escapeHtml(title)}</div>
      </div>
      <div class="modal-body" style="font-size: 13px; color: var(--text-medium); line-height: 1.5; padding: 14px 18px;">
        ${escapeHtml(message)}
      </div>
      <div class="modal-footer" style="display: flex; justify-content: flex-end; gap: 8px;">
        <button class="btn btn-secondary btn-sm" id="btn-modal-cancel">Cancel</button>
        <button class="btn ${isDestructive ? 'btn-danger' : 'btn-primary'} btn-sm" id="btn-modal-confirm">${escapeHtml(confirmLabel)}</button>
      </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    const cleanup = (result: boolean) => {
      overlay.classList.remove('open');
      setTimeout(() => {
        if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
      }, 150);
      resolve(result);
    };

    overlay.querySelector('#btn-modal-cancel')?.addEventListener('click', () => cleanup(false));
    overlay.querySelector('#btn-modal-confirm')?.addEventListener('click', () => cleanup(true));

    const keyListener = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        window.removeEventListener('keydown', keyListener);
        cleanup(false);
      } else if (e.key === 'Enter') {
        window.removeEventListener('keydown', keyListener);
        cleanup(true);
      }
    };
    window.addEventListener('keydown', keyListener);
  });
}

let activeContextMenu: HTMLElement | null = null;

export function dismissContextMenu() {
  if (activeContextMenu && activeContextMenu.parentNode) {
    activeContextMenu.parentNode.removeChild(activeContextMenu);
    activeContextMenu = null;
  }
}

interface MenuItem {
  label: string;
  shortcut?: string;
  icon?: string;
  disabled?: boolean;
  action: () => void;
}

/**
 * Initializes global desktop context menu handling, suppressing ugly webview default context menus.
 */
export function initNativeContextMenu() {
  window.addEventListener('contextmenu', (e: MouseEvent) => {
    e.preventDefault();
    dismissContextMenu();

    const target = e.target as HTMLElement | null;
    const selection = window.getSelection();
    const selectedText = selection ? selection.toString() : '';

    const isInput = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
    const inputElem = isInput ? (target as HTMLInputElement | HTMLTextAreaElement) : null;
    const isLogArea = target && (target.id === 'terminal-logs' || target.closest('#terminal-logs'));

    const items: (MenuItem | 'separator')[] = [];

    if (isInput && inputElem) {
      const hasSelection = (inputElem.selectionStart !== inputElem.selectionEnd);
      const isReadOnly = inputElem.readOnly || inputElem.disabled;

      items.push({
        label: 'Cut',
        shortcut: 'Ctrl+X',
        disabled: !hasSelection || isReadOnly,
        action: () => {
          document.execCommand('cut');
        }
      });
      items.push({
        label: 'Copy',
        shortcut: 'Ctrl+C',
        disabled: !hasSelection,
        action: () => {
          document.execCommand('copy');
        }
      });
      items.push({
        label: 'Paste',
        shortcut: 'Ctrl+V',
        disabled: isReadOnly,
        action: async () => {
          try {
            const text = await navigator.clipboard.readText();
            const start = inputElem.selectionStart || 0;
            const end = inputElem.selectionEnd || 0;
            inputElem.value = inputElem.value.substring(0, start) + text + inputElem.value.substring(end);
            inputElem.selectionStart = inputElem.selectionEnd = start + text.length;
            inputElem.dispatchEvent(new Event('input', { bubbles: true }));
          } catch {
            document.execCommand('paste');
          }
        }
      });
      items.push('separator');
      items.push({
        label: 'Select All',
        shortcut: 'Ctrl+A',
        action: () => {
          inputElem.select();
        }
      });
    } else if (isLogArea) {
      items.push({
        label: 'Copy Selected',
        shortcut: 'Ctrl+C',
        disabled: !selectedText,
        action: () => {
          if (selectedText) {
            navigator.clipboard.writeText(selectedText);
            showToast('Copied log selection to clipboard', 'info');
          }
        }
      });
      items.push({
        label: 'Copy All Logs',
        action: () => {
          const logElem = document.getElementById('terminal-logs');
          if (logElem && logElem.textContent) {
            navigator.clipboard.writeText(logElem.textContent);
            showToast('All console logs copied to clipboard', 'info');
          }
        }
      });
      items.push('separator');
      items.push({
        label: 'Clear Console',
        action: () => {
          const btnClear = document.getElementById('btn-clear-logs');
          if (btnClear) btnClear.click();
        }
      });
    } else if (selectedText) {
      items.push({
        label: 'Copy',
        shortcut: 'Ctrl+C',
        action: () => {
          navigator.clipboard.writeText(selectedText);
          showToast('Copied to clipboard', 'info');
        }
      });
      items.push('separator');
      items.push({
        label: 'Select All',
        shortcut: 'Ctrl+A',
        action: () => {
          document.execCommand('selectAll');
        }
      });
    } else {
      // General app menu
      items.push({
        label: 'Select All',
        shortcut: 'Ctrl+A',
        action: () => {
          document.execCommand('selectAll');
        }
      });
      items.push('separator');
      items.push({
        label: 'Run Backend Troubleshooter',
        action: () => {
          const btn = document.getElementById('btn-dash-troubleshoot');
          if (btn) btn.click();
        }
      });
      items.push({
        label: 'Open Settings',
        action: () => {
          const navBtn = document.querySelector('[data-tab="settings"]') as HTMLElement | null;
          if (navBtn) navBtn.click();
        }
      });
    }

    if (items.length === 0) return;

    // Render context menu element
    const menu = document.createElement('div');
    menu.className = 'native-context-menu';

    items.forEach((item) => {
      if (item === 'separator') {
        const sep = document.createElement('div');
        sep.className = 'native-context-separator';
        menu.appendChild(sep);
      } else {
        const row = document.createElement('div');
        row.className = `native-context-item ${item.disabled ? 'disabled' : ''}`;
        row.innerHTML = `
          <div class="native-context-label">
            <span>${escapeHtml(item.label)}</span>
          </div>
          ${item.shortcut ? `<span class="native-context-shortcut">${escapeHtml(item.shortcut)}</span>` : ''}
        `;

        if (!item.disabled) {
          row.addEventListener('click', (ev) => {
            ev.stopPropagation();
            dismissContextMenu();
            item.action();
          });
        }
        menu.appendChild(row);
      }
    });

    document.body.appendChild(menu);
    activeContextMenu = menu;

    // Viewport boundary adjustments
    const menuWidth = menu.offsetWidth || 200;
    const menuHeight = menu.offsetHeight || 160;

    let posX = e.clientX;
    let posY = e.clientY;

    if (posX + menuWidth > window.innerWidth) {
      posX = Math.max(10, window.innerWidth - menuWidth - 10);
    }
    if (posY + menuHeight > window.innerHeight) {
      posY = Math.max(10, window.innerHeight - menuHeight - 10);
    }

    menu.style.left = `${posX}px`;
    menu.style.top = `${posY}px`;
  });

  // Dismiss listeners
  window.addEventListener('click', () => dismissContextMenu());
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') dismissContextMenu();
  });
  window.addEventListener('resize', () => dismissContextMenu());
}

import { App, PluginSettingTab, Setting, TextComponent } from 'obsidian';
import ADRValidatorPlugin from '../main';

export interface ADRValidatorSettings {
    apiEndpoint: string;
    apiKey: string;
    additionalContext: string;
}

export const DEFAULT_SETTINGS: ADRValidatorSettings = {
    apiEndpoint: 'http://localhost:8000',
    apiKey: '',
    additionalContext: '',
};

export class ADRValidatorSettingTab extends PluginSettingTab {
    plugin: ADRValidatorPlugin;

    constructor(app: App, plugin: ADRValidatorPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        containerEl.createEl('h2', { text: 'ADR Security Validator' });
        containerEl.createEl('p', {
            text: 'Configure the connection to your ADR validation backend.',
            cls: 'adr-settings-desc'
        });

        new Setting(containerEl)
            .setName('API Endpoint')
            .setDesc('URL of your FastAPI backend')
            .addText(text => {
                text.setPlaceholder('http://localhost:8000')
                    .setValue(this.plugin.settings.apiEndpoint)
                    .onChange(async (value) => {
                        this.plugin.settings.apiEndpoint = value;
                        await this.plugin.saveSettings();
                    });
                (text as TextComponent).inputEl.style.width = '100%';
            });

        new Setting(containerEl)
            .setName('API Key')
            .setDesc('Authentication key (if required)')
            .addText(text => {
                text.setPlaceholder('sk-...')
                    .setValue(this.plugin.settings.apiKey)
                    .onChange(async (value) => {
                        this.plugin.settings.apiKey = value;
                        await this.plugin.saveSettings();
                    });
                (text as TextComponent).inputEl.type = 'password';
                (text as TextComponent).inputEl.style.width = '100%';
            });

        new Setting(containerEl)
            .setName('Additional Context')
            .setDesc('Extra information for the validator (e.g., tech stack)')
            .addTextArea(text => {
                text.setPlaceholder('Stack: Python, PostgreSQL, Docker, Kubernetes...')
                    .setValue(this.plugin.settings.additionalContext)
                    .onChange(async (value) => {
                        this.plugin.settings.additionalContext = value;
                        await this.plugin.saveSettings();
                    });
                (text as TextComponent).inputEl.style.width = '100%';
                (text as TextComponent).inputEl.rows = 3;
            });

        containerEl.createEl('h3', { text: 'Commands' });
        containerEl.createEl('ul', undefined, (ul) => {
            ul.createEl('li', { text: 'ADR Validator: Validate ADR - Validate current file' });
            ul.createEl('li', { text: 'ADR Validator: Index ADR - Add to Qdrant database' });
            ul.createEl('li', { text: 'ADR Validator: Check API Health - Test connection' });
        });
    }
}
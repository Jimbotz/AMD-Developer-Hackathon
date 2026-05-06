import { requestUrl, RequestUrlResponse } from 'obsidian';

export interface ADRValidationRequest {
    title: string;
    content: string;
    context?: string;
}

export interface SecurityRisk {
    severity: 'low' | 'medium' | 'high' | 'critical';
    type: string;
    description: string;
    code_example?: string;
    secure_alternative?: string;
}

export interface Contradiction {
    severity: 'low' | 'medium' | 'high';
    related_adr_title: string;
    description: string;
    source?: string;
}

export interface RelatedADR {
    title: string;
    similarity: number;
    status: string;
    category: string;
}

export interface ADRValidationResponse {
    status: 'approved' | 'needs_review' | 'needs_minor_revision' | 'rejected';
    message: string;
    contradictions: Contradiction[];
    security_risks: SecurityRisk[];
    recommendations: string[];
    related_adrs: RelatedADR[];
    detected_technologies: string[];
}

export interface ADRIndexRequest {
    title: string;
    content: string;
    metadata: {
        path: string;
        category: string;
        status: string;
    };
}

export class ADRAPIClient {
    constructor(
        private endpoint: string,
        private apiKey: string
    ) {}

    private getHeaders(): Record<string, string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        return headers;
    }

    async validateADR(request: ADRValidationRequest): Promise<ADRValidationResponse> {
        try {
            const response: RequestUrlResponse = await requestUrl({
                url: `${this.endpoint}/validate-adr`,
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify(request),
            });

            if (response.status !== 200) {
                throw new Error(`API returned ${response.status}`);
            }

            return response.json as ADRValidationResponse;
        } catch (error) {
            console.error('ADR validation error:', error);
            throw error;
        }
    }

    async indexADR(data: ADRIndexRequest): Promise<void> {
        try {
            const response: RequestUrlResponse = await requestUrl({
                url: `${this.endpoint}/index-adr`,
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify({
                    title: data.title,
                    content: data.content,
                    category: data.metadata.category,
                    status: data.metadata.status,
                    source: data.metadata.path,
                }),
            });

            if (response.status !== 200) {
                throw new Error(`API returned ${response.status}`);
            }
        } catch (error) {
            console.error('ADR indexing error:', error);
            throw error;
        }
    }

    async healthCheck(): Promise<boolean> {
        try {
            const response: RequestUrlResponse = await requestUrl({
                url: `${this.endpoint}/health`,
                method: 'GET',
                headers: this.getHeaders(),
            });

            return response.status === 200;
        } catch {
            return false;
        }
    }
}
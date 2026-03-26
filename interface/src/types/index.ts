export interface ExampleType {
    id: number;
    name: string;
    isActive: boolean;
}

export interface AppProps {
    title: string;
    items: ExampleType[];
}
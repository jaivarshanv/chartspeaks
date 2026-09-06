export interface ChartPoint {
  x: number
  y: number
}

export interface ChartSeries {
  name: string
  points: ChartPoint[]
}

export interface ChartAxis {
  label: string
  scaleType: 'linear' | 'logarithmic' | 'time'
  unit: string
  values: string[]
}

export interface ChartSemanticRepresentation {
  chartType: 'line' | 'bar' | 'scatter' | 'area' | 'unknown'

  title: string

  xAxis: ChartAxis

  yAxis: ChartAxis

  series: ChartSeries[]
}
export const testCSR: ChartSemanticRepresentation = {
  chartType: 'line',

  title: 'Test Sales Chart',

  xAxis: {
    label: 'Month',
    scaleType: 'linear',
    unit: '',
    values: ['1', '2', '3', '4', '5']
  },

  yAxis: {
    label: 'Sales',
    scaleType: 'linear',
    unit: 'units',
    values: ['10', '20', '30', '40', '50']
  },

  series: [
    {
      name: 'Sales',
      points: [
        { x: 1, y: 10 },
        { x: 2, y: 20 },
        { x: 3, y: 15 },
        { x: 4, y: 40 },
        { x: 5, y: 35 }
      ]
    }
  ]
}